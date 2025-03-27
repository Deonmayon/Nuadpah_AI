from flask import Flask
from flask_socketio import SocketIO
from config import Config
from services.socket_service import SocketService

def create_app():
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Initialize services
    socket_service = SocketService(socketio)
    
    # Register cleanup
    @app.teardown_appcontext
    def shutdown(exception=None):
        socket_service.cleanup()
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, host=Config.HOST, port=Config.SOCKETIO_PORT)