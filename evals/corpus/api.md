# API Reference Notes

## Authentication

API requests authenticate with a bearer token created in Settings, API keys.
Tokens are shown once at creation and cannot be retrieved afterwards. A token
inherits the permissions of the member who created it.

## Rate limits

The API allows 600 requests per minute per token on Team and 3000 on Enterprise.
Exceeding the limit returns HTTP 429 with a `Retry-After` header in seconds.
Rate limits are counted in fixed one-minute windows.

## Pagination

List endpoints return 50 items per page by default and 200 at most. Pagination
is cursor-based: the response carries `next_cursor`, and a missing `next_cursor`
means the last page.

## Webhooks

Webhook deliveries are retried with exponential backoff for up to 24 hours.
Each delivery carries an `X-Northwind-Signature` header, an HMAC-SHA256 of the
raw body using the endpoint's signing secret. A webhook endpoint that fails for
7 consecutive days is disabled automatically.

## Errors

Errors return a JSON body with `code`, `message`, and `request_id`. A 409 means
the resource was modified concurrently and the request should be retried with a
fresh version. Include `request_id` when contacting support.

## Versioning

The API version is pinned per token and defaults to the version current when the
token was created. Breaking changes ship as a new dated version; a version is
supported for at least 12 months after its successor is released.
