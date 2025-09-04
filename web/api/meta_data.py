from quart import Blueprint, jsonify, current_app

meta_data_bp = Blueprint("meta_data", __name__)


@meta_data_bp.route("/categories", methods=["GET"])
async def get_categories():
    return jsonify(current_app.meta_data_service.get_categories()), 200


@meta_data_bp.route("/search_paths", methods=["GET"])
async def get_search_paths():
    return jsonify(current_app.meta_data_service.get_search_paths()), 200
