# Admin Panel Setup

## 1. Run migrations

```bash
docker-compose exec web python manage.py makemigrations core
```

```bash
docker-compose exec web python manage.py migrate
```

## 2. Create admin user

```bash
docker-compose exec web python initialize_admin.py
```

## 3. Access

`http://localhost:8001/admin/`
