"""Samsung PRISM Version-Aware Demo Dataset.

A hand-authored, synthetic corpus of Samsung-style SDK snippets with version,
deprecation and replacement metadata. It is kept strictly separate from the CoIR
benchmark, is not an official Samsung dataset, and the APIs shown are illustrative.
"""

from typing import Dict, List, Optional, Tuple

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

def _doc(text: str, file: str, language: str, library: str, version: str, repository: str,
         deprecated: bool = False, replacement: Optional[str] = None) -> Dict:
    body = text.strip("\n")
    return {
        "text": "\n" + body + "\n",
        "metadata": {
            "library": library,
            "version": version,
            "deprecated": deprecated,
            "replacement": replacement,
            "file": file,
            "language": language,
            "repository": repository,
            "start_line": 1,
            "end_line": body.count("\n") + 1,
        },
    }


SAMSUNG_DEMO_CORPUS.update({
    # ---------------- Samsung Health: step count (v1 callback -> v2 coroutine) ----------------
    "health_steps_v1": _doc("""
// Samsung Health SDK v1 - read today's step count
@Deprecated("Blocking callback API. Use HealthDataStore.readData() suspend API in SDK v2")
fun readStepCountV1(store: HealthDataStore, callback: (Int) -> Unit) {
    val resolver = HealthDataResolver(store, null)
    val request = ReadRequest.Builder()
        .setDataType(StepCount.HEALTH_DATA_TYPE)
        .setLocalTimeRange(StepCount.START_TIME, StepCount.TIME_OFFSET, startOfDay(), now())
        .build()
    resolver.read(request).setResultListener { result ->
        callback(result.sumOf { it.getInt(StepCount.COUNT) })
    }
}
""", "health/v1/StepReader.kt", "kotlin", "samsung_health_sdk", "1.0", "samsung-health-android",
        deprecated=True, replacement="health_steps_v2"),
    "health_steps_v2": _doc("""
// Samsung Health SDK v2 - coroutine-based step count reader
suspend fun readTodaySteps(store: HealthDataStore): Long {
    val request = DataTypes.STEPS.readDataRequestBuilder
        .setLocalTimeFilter(LocalTimeFilter.of(LocalDate.now().atStartOfDay(), LocalDateTime.now()))
        .build()
    val response = store.readData(request)
    return response.dataList.sumOf { it.getValue(DataType.StepsType.COUNT) ?: 0L }
}
""", "health/v2/StepReader.kt", "kotlin", "samsung_health_sdk", "2.0", "samsung-health-android"),
    "health_heart_rate_v2": _doc("""
// Samsung Health SDK v2 - stream live heart rate from a paired Galaxy Watch
fun observeHeartRate(store: HealthDataStore): Flow<Int> = callbackFlow {
    val observer = HealthDataObserver { data ->
        data.lastOrNull()?.getValue(DataType.HeartRateType.HEART_RATE)?.let { trySend(it.toInt()) }
    }
    store.registerObserver(DataTypes.HEART_RATE, observer)
    awaitClose { store.unregisterObserver(observer) }
}
""", "health/v2/HeartRateMonitor.kt", "kotlin", "samsung_health_sdk", "2.0", "samsung-health-android"),
    "health_sleep_v2": _doc("""
// Samsung Health SDK v2 - aggregate sleep sessions for the last 7 days
suspend fun weeklySleepMinutes(store: HealthDataStore): Map<LocalDate, Long> {
    val request = DataTypes.SLEEP.readDataRequestBuilder
        .setLocalTimeFilter(LocalTimeFilter.since(LocalDateTime.now().minusDays(7)))
        .build()
    return store.readData(request).dataList
        .groupBy { it.startTime.atZone(ZoneId.systemDefault()).toLocalDate() }
        .mapValues { (_, sessions) -> sessions.sumOf { Duration.between(it.startTime, it.endTime).toMinutes() } }
}
""", "health/v2/SleepSummary.kt", "kotlin", "samsung_health_sdk", "2.0", "samsung-health-android"),
    "health_permissions_v2": _doc("""
// Samsung Health SDK v2 - request read permissions before accessing health data
suspend fun ensurePermissions(activity: Activity, store: HealthDataStore): Boolean {
    val wanted = setOf(
        Permission.of(DataTypes.STEPS, AccessType.READ),
        Permission.of(DataTypes.HEART_RATE, AccessType.READ),
        Permission.of(DataTypes.SLEEP, AccessType.READ),
    )
    val granted = store.getGrantedPermissions(wanted)
    if (granted.containsAll(wanted)) return true
    return store.requestPermissions(wanted, activity).containsAll(wanted)
}
""", "health/v2/Permissions.kt", "kotlin", "samsung_health_sdk", "2.0", "samsung-health-android"),

    # ---------------- Knox: device policy (v2 deprecated -> v3) ----------------
    "knox_camera_policy_v2": _doc("""
// Knox SDK v2 - disable the camera through RestrictionPolicy
/** @deprecated Knox 3.x moved device restrictions to KnoxDevicePolicy.setCameraState */
public boolean disableCameraV2(Context context) {
    EnterpriseDeviceManager edm = EnterpriseDeviceManager.getInstance(context);
    RestrictionPolicy policy = edm.getRestrictionPolicy();
    return policy.setCameraState(false);
}
""", "knox/v2/CameraPolicy.java", "java", "knox_sdk", "2.0", "knox-core-android",
        deprecated=True, replacement="knox_camera_policy_v3"),
    "knox_camera_policy_v3": _doc("""
// Knox SDK v3 - disable the camera with the unified device policy API
public PolicyResult disableCamera(Context context) {
    KnoxDevicePolicy policy = KnoxDevicePolicy.getInstance(context);
    PolicyResult result = policy.setCameraState(CameraState.DISABLED);
    if (!result.isSuccess()) {
        Log.w(TAG, "Camera restriction rejected: " + result.getErrorCode());
    }
    return result;
}
""", "knox/v3/CameraPolicy.java", "java", "knox_sdk", "3.0", "knox-core-android"),
    "knox_vpn_profile_v3": _doc("""
// Knox SDK v3 - provision a per-app VPN profile for managed apps
public void configurePerAppVpn(Context context, String profileName, List<String> packages) {
    KnoxVpnPolicy vpn = KnoxVpnPolicy.getInstance(context);
    JSONObject profile = new JSONObject()
        .put("profile_name", profileName)
        .put("vpn_type", "ipsec_ikev2_psk")
        .put("host", "vpn.corp.example.com");
    vpn.createVpnProfile(profile.toString());
    for (String pkg : packages) {
        vpn.addPackagesToVpn(profileName, new String[]{pkg});
    }
}
""", "knox/v3/VpnSetup.java", "java", "knox_sdk", "3.0", "knox-core-android"),
    "knox_attestation_v3": _doc("""
// Knox SDK v3 - verify device integrity with a server nonce (attestation)
public void requestAttestation(Context context, byte[] serverNonce, AttestationCallback cb) {
    KnoxAttestation attestation = KnoxAttestation.getInstance(context);
    attestation.startAttestation(serverNonce, blob -> {
        // Send the signed blob to the backend; never trust the verdict on-device.
        cb.onBlobReady(Base64.encodeToString(blob, Base64.NO_WRAP));
    });
}
""", "knox/v3/Attestation.java", "java", "knox_sdk", "3.0", "knox-core-android"),

    # ---------------- SmartThings: device commands (v2 REST deprecated -> v3 SDK) ----------------
    "st_device_command_v2": _doc("""
// SmartThings API v2 - send a switch command with raw REST calls
// @deprecated v2 endpoints are retired; use SmartThingsClient.devices.executeCommand (v3)
async function turnOnV2(token: string, deviceId: string): Promise<void> {
    await fetch(`https://graph.api.smartthings.com/api/devices/${deviceId}/on`, {
        method: "PUT",
        headers: { Authorization: `Bearer ${token}` },
    });
}
""", "src/legacy/commands_v2.ts", "typescript", "smartthings_sdk", "2.0", "smartthings-js-sdk",
        deprecated=True, replacement="st_device_command_v3"),
    "st_device_command_v3": _doc("""
// SmartThings SDK v3 - execute a capability command on a device
import { SmartThingsClient, BearerTokenAuthenticator } from "@smartthings/core-sdk";

export async function setSwitch(token: string, deviceId: string, on: boolean): Promise<void> {
    const client = new SmartThingsClient(new BearerTokenAuthenticator(token));
    await client.devices.executeCommand(deviceId, {
        component: "main",
        capability: "switch",
        command: on ? "on" : "off",
    });
}
""", "src/devices/commands.ts", "typescript", "smartthings_sdk", "3.0", "smartthings-js-sdk"),
    "st_webhook_verify_v3": _doc("""
// SmartThings SDK v3 - verify the HTTP signature of an incoming webhook (SmartApp)
import crypto from "crypto";

export function verifyWebhookSignature(rawBody: string, signature: string, secret: string): boolean {
    const expected = crypto.createHmac("sha256", secret).update(rawBody).digest("base64");
    const a = Buffer.from(expected);
    const b = Buffer.from(signature);
    return a.length === b.length && crypto.timingSafeEqual(a, b);
}
""", "src/smartapp/signature.ts", "typescript", "smartthings_sdk", "3.0", "smartthings-js-sdk"),
    "st_subscribe_events_v3": _doc("""
// SmartThings SDK v3 - subscribe a SmartApp to motion sensor events
export async function subscribeToMotion(context: SmartAppContext): Promise<void> {
    await context.api.subscriptions.delete();
    await context.api.subscriptions.subscribeToDevices(
        context.config.motionSensors, "motionSensor", "motion.active", "motionHandler",
    );
}

export async function motionHandler(context: SmartAppContext, event: DeviceEvent) {
    await context.api.devices.sendCommands(context.config.lights, "switch", "on");
}
""", "src/smartapp/subscriptions.ts", "typescript", "smartthings_sdk", "3.0", "smartthings-js-sdk"),
    "st_list_devices_v3": _doc("""
// SmartThings SDK v3 - list devices in a location filtered by capability
export async function listSwitches(client: SmartThingsClient, locationId: string) {
    const devices = await client.devices.list({ locationId, capability: "switch" });
    return devices.map(d => ({ id: d.deviceId, name: d.label ?? d.name, room: d.roomId }));
}
""", "src/devices/list.ts", "typescript", "smartthings_sdk", "3.0", "smartthings-js-sdk"),

    # ---------------- Wearables: Tizen (deprecated) -> Wear OS tiles ----------------
    "tizen_watch_app_v5": _doc("""
/* Tizen Native API 5.5 - Galaxy Watch app lifecycle (Tizen wearables are discontinued) */
/* @deprecated New Galaxy Watch apps target Wear OS; see wearos_tile_v1 */
static bool app_create(void *data) {
    appdata_s *ad = data;
    create_base_gui(ad);
    return true;
}
int main(int argc, char *argv[]) {
    appdata_s ad = {0,};
    ui_app_lifecycle_callback_s event_callback = {0,};
    event_callback.create = app_create;
    return ui_app_main(argc, argv, &event_callback, &ad);
}
""", "tizen/watch/main.c", "c", "tizen_native", "5.5", "galaxy-watch-legacy",
        deprecated=True, replacement="wearos_tile_v1"),
    "wearos_tile_v1": _doc("""
// Wear OS (Galaxy Watch) - a glanceable step-count Tile
class StepsTileService : SuspendingTileService() {
    override suspend fun resourcesRequest(requestParams: RequestBuilders.ResourcesRequest) =
        ResourceBuilders.Resources.Builder().setVersion("1").build()

    override suspend fun tileRequest(requestParams: RequestBuilders.TileRequest): TileBuilders.Tile {
        val steps = stepRepository.today()
        return TileBuilders.Tile.Builder()
            .setResourcesVersion("1")
            .setTileTimeline(TimelineBuilders.Timeline.fromLayoutElement(stepsLayout(steps)))
            .build()
    }
}
""", "wear/src/main/StepsTileService.kt", "kotlin", "wear_tiles", "1.0", "galaxy-watch-wearos"),

    # ---------------- In-app purchase (IAP v4 deprecated -> v6) ----------------
    "iap_purchase_v4": _doc("""
// Samsung In-App Purchase v4 - start a purchase
/** @deprecated IAP v4 is retired. Migrate to IapHelper.startPayment in IAP v6. */
public void buyItemV4(Activity activity, String itemId) {
    SamsungIapHelper helper = SamsungIapHelper.getInstance(activity, SamsungIapHelper.IAP_MODE_COMMERCIAL);
    helper.startPayment(itemId, true, (errorVo, purchaseVo) -> {
        if (errorVo.getErrorCode() == SamsungIapHelper.IAP_ERROR_NONE) deliver(purchaseVo);
    });
}
""", "iap/v4/Purchase.java", "java", "samsung_iap", "4.0", "galaxy-store-iap",
        deprecated=True, replacement="iap_purchase_v6"),
    "iap_purchase_v6": _doc("""
// Samsung In-App Purchase v6 - purchase with server-side receipt verification
public void buyItem(Activity activity, String itemId, String obfuscatedAccountId) {
    IapHelper iap = IapHelper.getInstance(activity);
    iap.setOperationMode(HelperDefine.OperationMode.OPERATION_MODE_PRODUCTION);
    iap.startPayment(itemId, obfuscatedAccountId, (error, purchase) -> {
        if (error.getErrorCode() == IapHelper.IAP_ERROR_NONE) {
            backend.verifyReceipt(purchase.getPurchaseId());   // verify before granting
        }
    });
}
""", "iap/v6/Purchase.java", "java", "samsung_iap", "6.0", "galaxy-store-iap"),
    "iap_consume_v6": _doc("""
// Samsung In-App Purchase v6 - consume a consumable item after it is delivered
public void consume(Activity activity, String purchaseId) {
    IapHelper.getInstance(activity).consumePurchasedItems(purchaseId, (error, results) -> {
        for (ConsumeVo vo : results) {
            Log.i(TAG, vo.getPurchaseId() + " consumed: " + vo.getStatusString());
        }
    });
}
""", "iap/v6/Consume.java", "java", "samsung_iap", "6.0", "galaxy-store-iap"),

    # ---------------- General engineering patterns used across Samsung apps ----------------
    "util_retry_backoff": _doc("""
# Retry a flaky network call with exponential backoff and jitter
import random, time

def retry_with_backoff(fn, retries: int = 5, base_delay: float = 0.5, max_delay: float = 8.0):
    for attempt in range(retries):
        try:
            return fn()
        except (ConnectionError, TimeoutError):
            if attempt == retries - 1:
                raise
            delay = min(max_delay, base_delay * 2 ** attempt)
            time.sleep(delay * random.uniform(0.5, 1.0))
""", "common/retry.py", "python", "common_utils", "1.0", "prism-shared-utils"),
    "util_token_refresh": _doc("""
# OAuth access token cache that refreshes shortly before expiry
import threading, time

class TokenProvider:
    def __init__(self, fetch_token, skew_seconds: int = 60):
        self._fetch, self._skew = fetch_token, skew_seconds
        self._token, self._expires_at = None, 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._token is None or time.time() >= self._expires_at - self._skew:
                payload = self._fetch()
                self._token = payload["access_token"]
                self._expires_at = time.time() + payload["expires_in"]
            return self._token
""", "common/auth/token_provider.py", "python", "common_utils", "1.0", "prism-shared-utils"),
    "util_rate_limiter": _doc("""
# Token-bucket rate limiter for outbound API requests
import time

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate, self.capacity = rate_per_sec, capacity
        self.tokens, self.updated = float(capacity), time.monotonic()

    def acquire(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
""", "common/ratelimit.py", "python", "common_utils", "1.0", "prism-shared-utils"),
    "util_lru_cache": _doc("""
// Thread-safe LRU cache backed by LinkedHashMap (access order)
public class LruCache<K, V> {
    private final Map<K, V> map;

    public LruCache(int capacity) {
        this.map = Collections.synchronizedMap(new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > capacity;
            }
        });
    }
    public V get(K key) { return map.get(key); }
    public void put(K key, V value) { map.put(key, value); }
}
""", "common/cache/LruCache.java", "java", "common_utils", "1.0", "prism-shared-utils"),
    "util_ble_scan": _doc("""
// Android BLE scan for nearby Galaxy accessories filtered by service UUID
@SuppressLint("MissingPermission")
fun scanForAccessories(context: Context, serviceUuid: UUID, onFound: (ScanResult) -> Unit): ScanCallback {
    val scanner = context.getSystemService(BluetoothManager::class.java).adapter.bluetoothLeScanner
    val filters = listOf(ScanFilter.Builder().setServiceUuid(ParcelUuid(serviceUuid)).build())
    val settings = ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build()
    val callback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) = onFound(result)
    }
    scanner.startScan(filters, settings, callback)
    return callback
}
""", "android/ble/AccessoryScanner.kt", "kotlin", "android_ble", "1.0", "galaxy-accessory-kit"),
    "util_paginate": _doc("""
# Iterate through every page of a cursor-paginated REST API
import requests

def iter_pages(url: str, headers: dict, page_size: int = 100):
    params = {"limit": page_size}
    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        yield from body.get("items", [])
        url = body.get("_links", {}).get("next", {}).get("href")
        params = None  # the next link already carries the cursor
""", "common/http/pagination.py", "python", "common_utils", "1.0", "prism-shared-utils"),
    "util_multipart_upload": _doc("""
// Upload a file with multipart/form-data using OkHttp
fun uploadLog(client: OkHttpClient, url: String, file: File): Boolean {
    val body = MultipartBody.Builder().setType(MultipartBody.FORM)
        .addFormDataPart("device", Build.MODEL)
        .addFormDataPart("file", file.name, file.asRequestBody("text/plain".toMediaType()))
        .build()
    client.newCall(Request.Builder().url(url).post(body).build()).execute().use { resp ->
        return resp.isSuccessful
    }
}
""", "android/net/LogUploader.kt", "kotlin", "android_net", "1.0", "galaxy-diagnostics"),
    "util_room_dao": _doc("""
// Room DAO for caching workout sessions offline
@Dao
interface WorkoutDao {
    @Query("SELECT * FROM workout WHERE start_time >= :since ORDER BY start_time DESC")
    fun observeSince(since: Long): Flow<List<WorkoutEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertAll(sessions: List<WorkoutEntity>)

    @Query("DELETE FROM workout WHERE start_time < :before")
    suspend fun prune(before: Long): Int
}
""", "android/db/WorkoutDao.kt", "kotlin", "androidx_room", "2.6", "samsung-health-android"),
    "util_csv_export": _doc("""
# Export health records to CSV with a stable column order
import csv
from typing import Iterable, Mapping

def export_csv(path: str, rows: Iterable[Mapping], columns: list[str]) -> int:
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count
""", "tools/export_csv.py", "python", "common_utils", "1.0", "prism-shared-utils"),
    "util_iso_datetime": _doc("""
# Parse ISO-8601 timestamps from device payloads into timezone-aware UTC datetimes
from datetime import datetime, timezone

def parse_device_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
""", "common/timeutil.py", "python", "common_utils", "1.0", "prism-shared-utils"),
    "util_json_parse_py": _doc("""
# Parse a JSON HTTP response safely and surface API error messages
import json

def parse_api_response(raw: bytes) -> dict:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON response: {exc}") from exc
    if body.get("status") != "ok":
        raise RuntimeError(body.get("error", {}).get("message", "unknown API error"))
    return body.get("data", {})
""", "common/http/json_response.py", "python", "common_utils", "1.0", "prism-shared-utils"),
})


SAMSUNG_DEMO_QUERIES: Dict[str, Dict] = {
    "sq1": {"text": "authentication using SDK version 3", "expected_doc": "samsung_auth_v3"},
    "sq2": {"text": "replacement for deprecated authentication API", "expected_doc": "samsung_auth_v3"},
    "sq3": {"text": "parse JSON response from a SmartThings device", "expected_doc": "samsung_json_parser_v3"},
    "sq4": {"text": "sensor streaming in Knox SDK version 2", "expected_doc": "samsung_sensor_stream_v2"},
    "sq5": {"text": "read today's step count with the current Samsung Health SDK", "expected_doc": "health_steps_v2"},
    "sq6": {"text": "migrate legacy v1 step count reader to the new API", "expected_doc": "health_steps_v2"},
    "sq7": {"text": "stream heart rate from Galaxy Watch", "expected_doc": "health_heart_rate_v2"},
    "sq8": {"text": "total sleep time over the past week", "expected_doc": "health_sleep_v2"},
    "sq9": {"text": "request health data permissions", "expected_doc": "health_permissions_v2"},
    "sq10": {"text": "disable camera with Knox device policy, latest version", "expected_doc": "knox_camera_policy_v3"},
    "sq11": {"text": "per-app VPN configuration for managed apps", "expected_doc": "knox_vpn_profile_v3"},
    "sq12": {"text": "verify device integrity attestation nonce", "expected_doc": "knox_attestation_v3"},
    "sq13": {"text": "turn on a SmartThings switch, replacement for the retired v2 REST endpoint",
             "expected_doc": "st_device_command_v3"},
    "sq14": {"text": "validate HMAC signature of incoming webhook", "expected_doc": "st_webhook_verify_v3"},
    "sq15": {"text": "turn on lights when motion is detected", "expected_doc": "st_subscribe_events_v3"},
    "sq16": {"text": "upgrade a Tizen Galaxy Watch app to the current platform", "expected_doc": "wearos_tile_v1"},
    "sq17": {"text": "in-app purchase with receipt verification", "expected_doc": "iap_purchase_v6"},
    "sq18": {"text": "migrate from IAP v4 startPayment", "expected_doc": "iap_purchase_v6"},
    "sq19": {"text": "retry with exponential backoff", "expected_doc": "util_retry_backoff"},
    "sq20": {"text": "refresh OAuth access token before it expires", "expected_doc": "util_token_refresh"},
    "sq21": {"text": "throttle outgoing API requests", "expected_doc": "util_rate_limiter"},
    "sq22": {"text": "scan for nearby bluetooth low energy devices", "expected_doc": "util_ble_scan"},
    "sq23": {"text": "iterate over all pages of a paginated REST API", "expected_doc": "util_paginate"},
    "sq24": {"text": "parse ISO 8601 timestamp to UTC", "expected_doc": "util_iso_datetime"},
}


def load_samsung_demo_data() -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Return (queries, corpus) for the Samsung version-aware demo."""
    return SAMSUNG_DEMO_QUERIES, SAMSUNG_DEMO_CORPUS
