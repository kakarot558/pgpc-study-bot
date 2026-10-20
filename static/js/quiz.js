// Flashcard quiz. Asks the local server for random cards (GET /api/flashcard); everything is
// inserted with textContent, and the score lives only in this page (nothing is stored anywhere).
(function () {
  "use strict";
  var subject = document.getElementById("quiz-subject");
  if (!subject) { return; }
  var q = document.getElementById("quiz-question"), a = document.getElementById("quiz-answer");
  var meta = document.getElementById("quiz-meta"), score = document.getElementById("quiz-score");
  var btn = { start: document.getElementById("quiz-start"), reveal: document.getElementById("quiz-reveal"),
              knew: document.getElementById("quiz-knew"), again: document.getElementById("quiz-again") };
  var seen = [], knew = 0, tried = 0, current = null;

  function setButtons(state) {            // state: idle | asking | revealed
    btn.reveal.disabled = state !== "asking";
    btn.knew.disabled = btn.again.disabled = state !== "revealed";
    btn.start.textContent = state === "idle" ? "Start" : "Next question";
    btn.start.disabled = state === "asking" || state === "revealed" && current === null;
  }
  function updateScore() { score.textContent = tried ? "Score this session: " + knew + " of " + tried + " known" : ""; }

  async function nextCard() {
    var url = "/api/flashcard?subject=" + encodeURIComponent(subject.value) + "&exclude=" + seen.join(",");
    try {
      var r = await fetch(url); var data = await r.json();
      if (!r.ok) { q.textContent = data.error || "Could not load a question."; return; }
      if (data.done) { current = null; q.textContent = "You have seen every card in this subject. Pick another subject or reload to restart."; a.hidden = true; meta.textContent = ""; setButtons("idle"); btn.start.disabled = true; return; }
      current = data; seen.push(data.id);
      q.textContent = data.question; a.textContent = data.answer; a.hidden = true;
      meta.textContent = data.topic + " | " + data.remaining + " more in this deck";
      setButtons("asking");
    } catch (err) { q.textContent = "Could not reach the local server."; }
  }
  function reveal() { a.hidden = false; setButtons("revealed"); }
  function grade(ok) { tried += 1; if (ok) { knew += 1; } updateScore(); nextCard(); }

  btn.start.addEventListener("click", nextCard);
  btn.reveal.addEventListener("click", reveal);
  btn.knew.addEventListener("click", function () { grade(true); });
  btn.again.addEventListener("click", function () { grade(false); });
  subject.addEventListener("change", function () { seen = []; current = null; q.textContent = "Press Start to get a question."; a.hidden = true; meta.textContent = ""; setButtons("idle"); btn.start.disabled = false; });
})();
