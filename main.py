from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates


app =FastAPI()
templates=Jinja2Templates(directory="templates")

@app.get("/", include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")

@app.get("/api/posts")
def get_posts(request: Request):
    return templates.TemplateResponse(request,"posts.html", { "posts": [{"id": 1, "title": "First Post"}, {"id": 2, "title": "Second Post"}]})