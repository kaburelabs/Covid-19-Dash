import time

import dash
import dash_html_components as html
from flask_socketio import SocketIO


app = dash.Dash(__name__)
server = app.server
server.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(server)

@socketio.on('welcome')
def handle_message(message):
    print(str(message))

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="//cdnjs.cloudflare.com/ajax/libs/socket.io/2.2.0/socket.io.js" integrity="sha256-yr4fRk/GU1ehYJPAs8P4JlTgu0Hdsp4ZKrx8bDEDC3I=" crossorigin="anonymous"></script>
        <script type="text/javascript" charset="utf-8">
            
            var socket = io();
            
            socket.on('connect', function() {
                socket.emit('hello', {data: 'connected'});
            });

            socket.on('update', function(data) {
                document.getElementById('finish').textContent=data+'/10';
            });
            
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    'Hello!',
    html.Button('Click me', id='trigger'),
    html.H1('', id='finish')
])


@app.callback(
    dash.dependencies.Output('finish', 'children'),
    [dash.dependencies.Input('trigger', 'n_clicks')])
def countdown(click):
    
    if not click:
        raise dash.exceptions.PreventUpdate()
    
    for i in range(10):
        socketio.emit('update', i)
        time.sleep(0.3)
    
    return click

if __name__ == '__main__':
    app.run_server(debug=True)