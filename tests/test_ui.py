from fastapi.testclient import TestClient


def test_the_question_page_is_served_without_hiding_the_api(client: TestClient) -> None:
    page = client.get("/")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert 'id="question"' in page.text
    assert client.get("/documents").headers["content-type"] == "application/json"
    assert client.get("/docs").status_code == 200


def test_the_page_never_parses_api_text_as_html(client: TestClient) -> None:
    # Document text reaches the page through answers and citations; it must stay text.
    assert "innerHTML" not in client.get("/").text
