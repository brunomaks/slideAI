### Installation setup

1. Activate python environment

2. Install packages from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

3. Create  `.env` file in the `main_app` service

    Add this to the contents of the created file:
    ```
    GRAYSCALE_URL='host-url/grayscale/'
    FLIP_URL='host-url/flip/'

    HOST_URL='host-url'
    ```

4. Run Django server:
    ```
    python manage.py migrate
    python manage.py runserver 8001
    python -m uvicorn hand_gesture_recognition.asgi:application --host 0.0.0.0 --port 8001 --reload
    ```
