"""Reference solution for the ag_todoapp agent task.

Exists so the grader can be trusted: if a correct implementation does not score
every check, the fault is in the checks rather than in the model being measured.
"""
from flask import Flask, jsonify, render_template_string, request

from . import storage

__all__ = ["create_app", "storage"]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Tasks</title>
<style>
 body { font: 16px/1.5 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; }
 li { list-style: none; display: flex; gap: .75rem; align-items: baseline;
      padding: .6rem; border-left: .4rem solid; margin-bottom: .4rem; cursor: grab; }
 li.done .title { text-decoration: line-through; opacity: .55; }
 .desc { color: #666; font-size: .9em; }
</style></head><body>
<h1>Tasks</h1>
<form id="add">
  <input name="title" placeholder="Title" required>
  <input name="description" placeholder="Description">
  <input name="color" type="color" value="#3366cc">
  <button type="submit">Add</button>
</form>
<ul id="list">
{% for t in tasks %}
  <li draggable="true" data-id="{{ t.id }}" style="border-color: {{ t.color }}"
      class="{{ 'done' if t.done else '' }}">
    <input type="checkbox" {{ 'checked' if t.done else '' }}>
    <span class="title">{{ t.title }}</span>
    <span class="desc">{{ t.description }}</span>
  </li>
{% endfor %}
</ul>
<script>
const list = document.getElementById('list');
let held = null;
list.addEventListener('dragstart', e => held = e.target.closest('li'));
list.addEventListener('dragover', e => {
  e.preventDefault();
  const over = e.target.closest('li');
  if (over && held && over !== held) {
    const after = over.getBoundingClientRect().top + over.offsetHeight / 2 < e.clientY;
    list.insertBefore(held, after ? over.nextSibling : over);
  }
});
list.addEventListener('drop', async e => {
  e.preventDefault();
  const order = [...list.children].map(li => Number(li.dataset.id));
  await fetch('/api/tasks/reorder', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify({order})});
});
list.addEventListener('change', async e => {
  const li = e.target.closest('li');
  await fetch('/api/tasks/' + li.dataset.id, {method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({done: e.target.checked})});
  location.reload();
});
list.addEventListener('dblclick', async e => {
  const span = e.target.closest('.title');
  if (!span) return;
  const li = span.closest('li');
  const value = prompt('Edit title', span.textContent);
  if (value === null || value === '') return;
  await fetch('/api/tasks/' + li.dataset.id, {method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title: value})});
  location.reload();
});
document.getElementById('add').addEventListener('submit', async e => {
  e.preventDefault();
  const f = new FormData(e.target);
  await fetch('/api/tasks', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.fromEntries(f))});
  location.reload();
});
</script>
</body></html>"""


def create_app():
    app = Flask(__name__)

    def flag(name):
        raw = request.args.get(name)
        return None if raw is None else raw.lower() in ("1", "true", "yes")

    @app.get("/")
    def index():
        return render_template_string(PAGE, tasks=storage.listing())

    @app.get("/api/tasks")
    def api_list():
        return jsonify(storage.listing(done=flag("done"), query=request.args.get("q")))

    @app.post("/api/tasks")
    def api_create():
        body = request.get_json(silent=True) or {}
        if not body.get("title"):
            return jsonify(error="title is required"), 400
        return jsonify(storage.create(body["title"], body.get("description", ""),
                                      body.get("color", "#888888"))), 201

    @app.get("/api/tasks/<int:task_id>")
    def api_get(task_id):
        task = storage.get(task_id)
        return (jsonify(task), 200) if task else (jsonify(error="not found"), 404)

    @app.patch("/api/tasks/<int:task_id>")
    def api_update(task_id):
        task = storage.update(task_id, request.get_json(silent=True) or {})
        return (jsonify(task), 200) if task else (jsonify(error="not found"), 404)

    @app.delete("/api/tasks/<int:task_id>")
    def api_delete(task_id):
        if not storage.delete(task_id):
            return jsonify(error="not found"), 404
        return "", 204

    @app.post("/api/tasks/reorder")
    def api_reorder():
        body = request.get_json(silent=True) or {}
        return jsonify(storage.reorder(body.get("order") or []))

    return app
