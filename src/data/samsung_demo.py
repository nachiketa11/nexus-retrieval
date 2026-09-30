"""Samsung PRISM Version-Aware Demo Dataset.

Kept strictly separate from the CoIR benchmark.
"""

from typing import Dict, List, Tuple

SAMSUNG_DEMO_CORPUS: Dict[str, Dict] = {
    "samsung_auth_v1": {
        "text": """
# Samsung Health SDK v1 Authentication
def authenticate_user_v1(api_key, secret):
    \"\"\"[DEPRECATED] Authenticate user with legacy API key method.
    Use authenticate_oauth_v3 in SDK v3 instead.
    \"\"\"
    import hashlib
    signature = hashlib.sha256(f"{api_key}:{secret}".encode()).hexdigest()
    return {"status": "success", "token": signature, "version": "1.0"}
""",
        "metadata": {
            "library": "samsung_health_sdk",
            "version": "1.0",
            "deprecated": True,
            "replacement": "samsung_auth_v3",
            "file": "sdk/v1/auth.py",
            "language": "python",
            "repository": "samsung-health-android",
            "start_line": 1,
            "end_line": 10,
        },
    },
    "samsung_auth_v2": {
        "text": """
// Samsung Health SDK v2 Authentication
public class AuthenticationV2 {
    /**
     * Authenticate via OAuth 2.0 Client Credentials.
     * @deprecated Use SamsungAuthSDK v3.0 PKCE auth flow instead.
     */
    public static AuthToken authenticateClient(String clientId, String clientSecret) {
        System.out.println("Authenticating SDK v2...");
        return new AuthToken("token_v2_oauth", 3600);
    }
}
""",
        "metadata": {
            "library": "samsung_health_sdk",
            "version": "2.0",
            "deprecated": True,
            "replacement": "samsung_auth_v3",
            "file": "sdk/v2/AuthenticationV2.java",
            "language": "java",
            "repository": "samsung-health-android",
            "start_line": 1,
            "end_line": 12,
        },
    },
    "samsung_auth_v3": {
        "text": """
# Samsung Health SDK v3 Authentication
from samsung_auth import OAuth3Client

def authenticate_oauth_v3(client_id: str, scope: list[str]) -> dict:
    \"\"\"Authenticate using SDK version 3 PKCE Flow.
    Recommended replacement for deprecated v1 and v2 auth methods.
    \"\"\"
    client = OAuth3Client(client_id=client_id, use_pkce=True)
    token = client.acquire_token(scopes=scope)
    return {
        "access_token": token.secret,
        "expires_in": token.expires_in,
        "token_type": "Bearer",
        "version": "3.0"
    }
""",
        "metadata": {
            "library": "samsung_health_sdk",
            "version": "3.0",
            "deprecated": False,
            "replacement": None,
            "file": "sdk/v3/auth_pkce.py",
            "language": "python",
            "repository": "samsung-health-android",
            "start_line": 1,
            "end_line": 16,
        },
    },
    "samsung_sensor_stream_v2": {
        "text": """
// Samsung Knox Sensor Streaming API v2
#include <iostream>
#include <vector>

class KnoxSensorStream {
public:
    // Streaming sensor data in Knox SDK version 2
    void startStream(int sensorId, int sampleRateHz) {
        std::cout << "Streaming sensor " << sensorId << " at " << sampleRateHz << "Hz" << std::endl;
    }
};
""",
        "metadata": {
            "library": "knox_sensor_sdk",
            "version": "2.0",
            "deprecated": False,
            "replacement": None,
            "file": "knox/sensor/stream.cpp",
            "language": "cpp",
            "repository": "knox-core-native",
            "start_line": 1,
            "end_line": 14,
        },
    },
    "samsung_json_parser_v3": {
        "text": """
// Samsung SmartThings Response JSON Parser
export function parseSmartThingsResponse(rawJson: string): Record<string, any> {
    try {
        const parsed = JSON.parse(rawJson);
        if (parsed.status !== "OK") {
            throw new Error(`Device response error: ${parsed.error_message}`);
        }
        return parsed.payload;
    } catch (e) {
        console.error("Failed to parse JSON response:", e);
        throw e;
    }
}
""",
        "metadata": {
            "library": "smartthings_sdk",
            "version": "3.0",
            "deprecated": False,
            "replacement": None,
            "file": "src/parser/json_parser.ts",
            "language": "typescript",
            "repository": "smartthings-js-sdk",
            "start_line": 1,
            "end_line": 14,
        },
    },
}

SAMSUNG_DEMO_QUERIES: Dict[str, Dict] = {
    "sq1": {
        "text": "authentication using SDK version 3",
        "expected_doc": "samsung_auth_v3",
    },
    "sq2": {
        "text": "replacement for deprecated authentication API",
        "expected_doc": "samsung_auth_v3",
    },
    "sq3": {
        "text": "parse JSON response",
        "expected_doc": "samsung_json_parser_v3",
    },
    "sq4": {
        "text": "sensor streaming in Knox SDK version 2",
        "expected_doc": "samsung_sensor_stream_v2",
    },
}


def load_samsung_demo_data() -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Return (queries, corpus) for the Samsung version-aware demo."""
    return SAMSUNG_DEMO_QUERIES, SAMSUNG_DEMO_CORPUS
