# Use a standard Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies (including gdown)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- This section acts as your build command ---
# 1. Create the directory for your models
RUN mkdir -p ./model_artifacts

# 2. Download each model file from Google Drive
RUN gdown -O ./model_artifacts/tfidf_sim.npy '1_OfQgpaUMc7RmuiquFjfhCLnrtk3Mabw'
RUN gdown -O ./model_artifacts/content_artifacts.pkl '1fIkGucbRXixxgWr8AtTipj5XHMUe09OP'
RUN gdown -O ./model_artifacts/best_model_state_dict.pth '1-JXbcZNEVGOWcT7w8bIftYo0j40TZBMk'
RUN gdown -O ./model_artifacts/model_artifacts.pkl '1mfeMxxfpeurFDzFbVUvVhwEwwSP9xNpj'
# ----------------------------------------------

# Copy the rest of your application code
COPY . /code/

# Tell the container what command to run when it starts
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]