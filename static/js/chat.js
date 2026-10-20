// StudyBuddy chat page. Talks ONLY to this Flask server (POST /chat).
// All text is inserted with textContent, so user input can never become HTML.
(function () {
  "use strict";
  var form = document.getElementById("chat-form");
  if (!form) { return; }
  var input = document.getElementById("chat-input");
  var sendBtn = document.getElementById("send-btn");
  var clearBtn = document.getElementById("clear-btn");
  var box = document.getElementById("messages");
  var status = document.getElementById("status");
  var maxLength = parseInt(form.dataset.maxLength, 10) || 300;
  var WELCOME = "Hi! I am StudyBuddy. I answer BSCS (Computer Science) subject questions. Pick a suggested question or type your own.";
  var busy = false;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  function addBubble(kind, text, extraClass) {
    var bubble = el("div", "bubble " + kind + (extraClass ? " " + extraClass : ""));
    bubble.appendChild(el("p", "", text));
    box.appendChild(bubble);
    box.scrollTop = box.scrollHeight;
    return bubble;
  }

  function addTyping() {
    var bubble = el("div", "bubble bot typing");
    bubble.setAttribute("aria-label", "StudyBuddy is thinking");
    for (var i = 0; i < 3; i++) { bubble.appendChild(el("span")); }
    box.appendChild(bubble);
    box.scrollTop = box.scrollHeight;
    return bubble;
  }

  function setBusy(value) {
    busy = value;
    sendBtn.disabled = value;
    sendBtn.textContent = value ? "Thinking..." : "Send";
    input.disabled = value;
    if (!value) { input.focus(); }
  }

  function showBotAnswer(data, userText) {
    var bubble = addBubble("bot", data.answer, data.match_found ? "" : "fallback");
    var parts = [];
    if (data.match_found) {
      parts.push("Topic: " + data.topic);
      parts.push("Matched question: " + data.matched_question);
    }
    parts.push("Similarity: " + Number(data.similarity_score).toFixed(2) + " (word-overlap score, not a probability)");
    if (data.match_found && data.confidence === "low") {
      bubble.appendChild(el("p", "meta caution", "This is the closest match I found. It may not answer your exact question, so please check it."));
    }
    bubble.appendChild(el("p", "meta", parts.join("  |  ")));
    var offered = [];
    if (!data.match_found && data.suggestion) { offered.push(data.suggestion); }
    (data.related_questions || []).forEach(function (q) { offered.push(q); });
    if (offered.length) {
      bubble.appendChild(el("p", "meta", data.match_found ? "Related questions:" : "Did you mean one of these? Tap to ask it:"));
      offered.forEach(function (q) {
        var b = el("button", "hint-btn related-btn", q);
        b.type = "button";
        b.addEventListener("click", function () { sendMessage(q); });
        bubble.appendChild(b);
      });
    }
    addFeedback(bubble, userText, data);
    box.scrollTop = box.scrollHeight;
  }

  // "Was this helpful?" buttons. The rating is saved in a local file on this computer only.
  function addFeedback(bubble, userText, data) {
    var row = el("p", "meta feedback");
    row.appendChild(document.createTextNode("Was this helpful? "));
    [["up", "Helpful"], ["down", "Not helpful"]].forEach(function (pair) {
      var b = el("button", "hint-btn feedback-btn", pair[1]);
      b.type = "button";
      b.setAttribute("aria-label", pair[1] + " answer");
      b.addEventListener("click", async function () {
        row.textContent = "Saving...";
        try {
          var r = await fetch("/feedback", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userText, rating: pair[0], matched_question: data.matched_question,
                                   similarity_score: data.similarity_score, match_found: data.match_found })
          });
          row.textContent = r.ok ? "Thanks! Your feedback is saved on this computer only." : "Sorry, feedback could not be saved.";
        } catch (err) { row.textContent = "Sorry, feedback could not be saved."; }
      });
      row.appendChild(b);
      row.appendChild(document.createTextNode("  "));
    });
    bubble.appendChild(row);
  }

  async function sendMessage(text) {
    text = text.trim();
    if (!text || busy) { return; }
    addBubble("user", text);
    input.value = "";
    setBusy(true);
    status.textContent = "Looking for an answer";
    var typing = addTyping();
    try {
      var response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });
      var data = await response.json().catch(function () { return {}; });
      typing.remove();
      if (!response.ok) {
        addBubble("bot", data.error || "Sorry, something went wrong. Please try again.", "error");
      } else {
        showBotAnswer(data, text);
      }
    } catch (err) {
      typing.remove();
      addBubble("bot", "I could not reach the local server. Is the app still running?", "error");
    } finally {
      status.textContent = "";
      setBusy(false);
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    sendMessage(input.value);
  });

  clearBtn.addEventListener("click", function () {
    box.textContent = "";
    addBubble("bot", WELCOME);
    input.focus();
  });

  // Suggested questions
  document.querySelectorAll(".suggestion").forEach(function (button) {
    button.addEventListener("click", function () { sendMessage(button.dataset.question); });
  });

  // Topic filter chips
  var chips = document.querySelectorAll("#topic-filters .chip");
  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      var topic = chip.dataset.topic;
      chips.forEach(function (c) {
        var active = c === chip;
        c.classList.toggle("is-active", active);
        c.setAttribute("aria-pressed", active ? "true" : "false");
      });
      document.querySelectorAll("#suggestions li").forEach(function (li) {
        li.hidden = Boolean(topic) && li.dataset.topic !== topic;
      });
    });
  });

  // Links from the Topics page look like /?q=What+is+recall%3F : prefill the box (not auto-send).
  var preset = new URLSearchParams(window.location.search).get("q");
  if (preset) { input.value = preset.slice(0, maxLength); }
})();
