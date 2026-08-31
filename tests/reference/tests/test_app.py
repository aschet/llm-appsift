"""The tests a model is expected to write for itself, written here instead.

They stand in for the model's own suite when the reference is graded, so the
`own_tests` check is exercised by the same script that grades a real run.

Written with pytest, unlike every other test in this repository, which uses
unittest and nothing outside the standard library. That is deliberate rather than
inconsistent: this file is not part of this project's suite, it is the graded
artefact's, and the grader runs a model's tests by shelling out to pytest. Models
write pytest, so writing this the same way exercises the path that will actually
be taken. It is excluded from collection here and reached only through
codesift.agent.verify.
"""
import pytest

from todoapp import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TODO_DB", str(tmp_path / "t.db"))
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_the_store_starts_with_three_examples(client):
    tasks = client.get("/api/tasks").get_json()
    assert len(tasks) == 3
    assert [t["position"] for t in tasks] == sorted(t["position"] for t in tasks)


def test_a_created_task_is_open_and_last(client):
    made = client.post("/api/tasks", json={"title": "New", "description": "d",
                                           "color": "#010203"})
    assert made.status_code == 201
    assert made.get_json()["done"] is False
    assert client.get("/api/tasks").get_json()[-1]["id"] == made.get_json()["id"]


def test_a_patch_leaves_untouched_fields_alone(client):
    made = client.post("/api/tasks", json={"title": "Keep", "description": "mine",
                                           "color": "#010203"}).get_json()
    after = client.patch(f"/api/tasks/{made['id']}", json={"done": True}).get_json()
    assert after["done"] is True and after["description"] == "mine"


def test_a_deleted_task_is_gone(client):
    made = client.post("/api/tasks", json={"title": "Bye"}).get_json()
    assert client.delete(f"/api/tasks/{made['id']}").status_code == 204
    assert client.get(f"/api/tasks/{made['id']}").status_code == 404


def test_filters_narrow_the_listing(client):
    made = client.post("/api/tasks", json={"title": "Xylophone"}).get_json()
    client.patch(f"/api/tasks/{made['id']}", json={"done": True})
    assert [t["id"] for t in client.get("/api/tasks?done=true").get_json()] == [made["id"]]
    assert len(client.get("/api/tasks?done=false").get_json()) == 3
    assert len(client.get("/api/tasks?q=xylo").get_json()) == 1


def test_reordering_survives_a_new_connection(client):
    ids = [t["id"] for t in client.get("/api/tasks").get_json()]
    client.post("/api/tasks/reorder", json={"order": ids[::-1]})
    assert [t["id"] for t in client.get("/api/tasks").get_json()] == ids[::-1]


def test_the_page_renders_every_task(client):
    page = client.get("/").get_data(as_text=True)
    for task in client.get("/api/tasks").get_json():
        assert task["title"] in page
    assert "draggable" in page
