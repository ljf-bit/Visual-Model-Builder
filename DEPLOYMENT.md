# Deployment

This project has a static Vite frontend and a FastAPI backend. To make it
available through a public link, deploy both parts and connect the frontend to
the backend URL.

## Backend

Deploy `visual-model-builder/backend` as a Docker web service.

The backend Dockerfile is included at:

```text
visual-model-builder/backend/Dockerfile
```

For Render Blueprint-style deployment, `render.yaml` is included at the
repository root. After the backend service is created, set:

```text
VMB_CORS_ORIGINS=https://your-frontend-domain.example.com
```

For temporary preview deployments, you can also use:

```text
VMB_CORS_ORIGIN_REGEX=https://.*\.vercel\.app
```

Use the backend service URL plus `/health` to verify that the API is running.
Use `/docs` on the backend service URL to view the FastAPI documentation.

## Frontend

Deploy `visual-model-builder/frontend` as a Vite static site.

Set this frontend environment variable before building:

```text
VITE_API_BASE_URL=https://your-backend-service.example.com
```

Build command:

```text
npm run build
```

Output directory:

```text
dist
```

## Recommended Order

1. Deploy the backend and confirm `/health` works.
2. Deploy the frontend with `VITE_API_BASE_URL` pointing to the backend URL.
3. Add the final frontend domain to backend `VMB_CORS_ORIGINS`.
4. Redeploy the backend after changing CORS settings.

## Notes

- Do not commit local datasets, model weights, or generated training runs.
- MNIST is downloaded on first use if it is not already present in the backend
  runtime.
- Long CPU training jobs may need a larger backend instance than a small free
  service tier.
