"""Kalshi API authentication with RSA request signing.

Kalshi uses RSA-based request signing for API authentication.
Each request must include:
- KALSHI-ACCESS-KEY: Your API key ID
- KALSHI-ACCESS-SIGNATURE: RSA signature of the request
- KALSHI-ACCESS-TIMESTAMP: Unix timestamp in milliseconds
"""

import base64
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from footbe_trader.common.config import KalshiConfig


@dataclass
class SignedRequest:
    """A signed request ready to send to Kalshi."""

    headers: dict[str, str]
    timestamp_ms: int


class KalshiAuth:
    """Kalshi API authentication handler.

    Implements RSA request signing per Kalshi API docs.
    """

    def __init__(self, config: KalshiConfig):
        """Initialize auth handler.

        Args:
            config: Kalshi configuration with API key and private key.
        """
        self.config = config
        self._private_key: rsa.RSAPrivateKey | None = None

    @property
    def private_key(self) -> rsa.RSAPrivateKey:
        """Load and cache the RSA private key.

        Returns:
            RSA private key object.

        Raises:
            ValueError: If private key is invalid.
        """
        if self._private_key is None:
            key_content = self.config.get_private_key_content()
            self._private_key = serialization.load_pem_private_key(
                key_content.encode(),
                password=None,
            )
            if not isinstance(self._private_key, rsa.RSAPrivateKey):
                raise ValueError("Private key must be RSA")
        return self._private_key

    def sign_request(
        self,
        method: str,
        path: str,
        timestamp_ms: int | None = None,
    ) -> SignedRequest:
        """Sign a request for Kalshi API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (e.g., /trade-api/v2/portfolio/balance)
            timestamp_ms: Optional timestamp in milliseconds (uses current time if not provided)

        Returns:
            SignedRequest with headers to include in the request.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Build the message to sign: timestamp + method + path
        # Per Kalshi docs: the signature is over "{timestamp}{method}{path}"
        message = f"{timestamp_ms}{method.upper()}{path}"

        # Sign with RSA-PSS (as per Kalshi's official Python starter code)
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        # Base64 encode the signature
        signature_b64 = base64.b64encode(signature).decode()

        return SignedRequest(
            headers={
                "KALSHI-ACCESS-KEY": self.config.api_key_id,
                "KALSHI-ACCESS-SIGNATURE": signature_b64,
                "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
                "Content-Type": "application/json",
            },
            timestamp_ms=timestamp_ms,
        )

    def validate_config(self) -> tuple[bool, str]:
        """Validate that auth configuration is complete.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not self.config.api_key_id:
            return False, "KALSHI_API_KEY_ID not configured"

        try:
            _ = self.private_key
        except ValueError as e:
            return False, f"Invalid private key: {e}"
        except FileNotFoundError as e:
            return False, str(e)

        return True, ""
