from fasthtml.common import *
from claudette import *
import asyncio

# Set up the app, including daisyui and tailwind for the chat component
tlink = Script(src="https://cdn.tailwindcss.com")
dlink = Link(rel="stylesheet",
             href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
app = FastHTML(hdrs=(tlink, dlink, picolink), ws_hdr=True)
rt = app.route


@app.route("/")
def get():
    consent_form = Form(
        Checkbox(
            id="consent",
            name="consent",
            label="I consent to participate in this experiment"),
        Button("Proceed to Experiment", id="proceed-btn", disabled=True),
        hx_post="/experiment",
        hx_trigger="click from:#proceed-btn"
    )

    script = Script("""
    document.getElementById('consent').addEventListener('change', function() {
        document.getElementById('proceed-btn').disabled = !this.checked;
    });
    """)

    return Titled("Experiment Consent", consent_form, script)


if __name__ == '__main__': uvicorn.run("ws_streaming:app", host='0.0.0.0', port=8000, reload=True)