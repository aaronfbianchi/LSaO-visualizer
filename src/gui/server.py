from flask import Flask, render_template, request

app = Flask(__name__)

def process_audio():
  print("Python function executed: Processing audio data...")
  # Put your heavy lifting, ffmpeg calls, or data processing here!
  return "Audio processed successfully by Python!"


@app.route("/", methods=["GET", "POST"])
def index():
  message = ""
  if request.method == "POST":
    # Check which button was clicked based on its 'name' or 'value'
    action = request.form.get("action")

    if action == "run_process":
      # Call your Python function when the button is pressed
      message = process_audio()

  return render_template("index.html", message=message)