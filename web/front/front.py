from quart import Blueprint, current_app, render_template

front_bp = Blueprint(
    "front",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/front/static",
)

@front_bp.route("/", methods=["GET"])
async def index():
    app_name = current_app.app_name
    app_description = current_app.app_description
    return await render_template("index.html",
                                 app_name=app_name,
                                 app_description=app_description)

@front_bp.route("/document_manager", methods=["GET"])
async def document_manager():
    app_name = current_app.app_name
    app_description = current_app.app_description
    return await render_template("document_manager.html",
                                 app_name=app_name,
                                 app_description=app_description)
