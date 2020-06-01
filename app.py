from flask import request, redirect, render_template
from flask import Flask
import model
import os


app = Flask(__name__, template_folder='templates/')
app.config["IMAGE_UPLOADS"] = "static\img"

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]
            path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            image.save(path)
            print(path)
            result = model.predict_class(path)
            os.remove(path)
            return result

    return render_template("upload.html")

if __name__ == '__main__':
    app.run()