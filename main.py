import functools
from fasthtml.common import *
from starlette.websockets import WebSocket
from starlette.endpoints import WebSocketEndpoint
import fastwebrl
import asyncio
#from fastwebrl.managers import ExperimentManager
from fastwebrl.manager import ExperimentManager
#from fastwebrl.managers import WebSocketHandler


class WebSocketHandler(WebSocketEndpoint):
    encoding = 'json'

    async def on_connect(self, websocket):
        await websocket.accept()
        print('on_connect')

    async def on_receive(self, websocket: WebSocket, data):
        print('on_receive', data)
        #import ipdb; ipdb.set_trace()
        await websocket.send(Div(
            H1("replacement"),
            id='stateImageContainer',
        ))

    async def on_disconnect(self, websocket, close_code):
        print('on_disconnect')

import experiment_1
import experiment_test


tlink = Script(src="https://cdn.tailwindcss.com")
dlink = Link(rel="stylesheet",
             href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
default_style = Style(
    open(fastwebrl.default_css_file()).read(),
    type="text/css", rel="stylesheet")

secret_key = "your-secret-key-here"
app = FastHTML(
    hdrs=[default_style],
    ws_hdr=True,
    middleware=[
        Middleware(SessionMiddleware, secret_key=secret_key)
    ],
)

rt = app.route

stages = experiment_test.stages

manager = ExperimentManager(
    stages=stages,
)


@rt("/")
async def get(request: Request):
    print('loadin?')
    session = manager.load_session(request)
    return manager.load_stage(request)

@app.post("/experiment")
async def start_experiment(request: Request):
    stage_idx = manager.get_stage_idx(request)
    assert stage_idx == 0, 'this should only be called from initial page'
    manager.increment_stage(request)
    return manager.load_stage(request)


#async def on_connect(send): print('connect')
#async def on_disconnect(send): print('disconnet')

#@app.ws('/wscon', conn=on_connect, disconn=on_disconnect)
#async def ws(msg: str, send):
#    print('wscon')
#    await asyncio.sleep(1)
#    print('wscon')

app.add_websocket_route("/ws", WebSocketHandler)
#app.add_websocket_route("/ws",
#                        functools.partial(WebSocketHandler, manager=manager))

if __name__ == '__main__':
   import uvicorn
   uvicorn.run("ws:app", host='0.0.0.0', port=int(
       os.environ.get("PORT", 8003)), reload=True)
