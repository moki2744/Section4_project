from flask import Flask

def create_app():
    app = Flask(__name__)

    from project_app.views.main_views import main_bp
    from project_app.views.result_views import result_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(result_bp)
    return app

if __name__ == "__main__":
  app = create_app()
  app.run()