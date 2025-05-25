package com.example.tradingapp;

import MOFSLOPENAPI.CMOFSLOPENAPI;
import MOFSLOPENAPI.Client.PlaceOrderinfo;
import org.json.JSONObject; 
// import org.json.JSONException; // Not explicitly thrown by placeholder, but good for real SDKs

// import java.util.Scanner; // No longer needed here, moved to Main

/**
 * Service class to interact with the MOFSL Open API.
 * This class encapsulates the logic for initializing the API client,
 * handling login (including OTP verification), placing orders, and
 * potentially other API interactions. It relies on {@link AppConfig}
 * for configuration settings.
 */
public class ApiService {

    private CMOFSLOPENAPI MofslApi;
    private String authToken;
    private String clientCodeForSession; // Store client code obtained after login

    /**
     * Constructor for use in testing, allowing injection of a mock or custom CMOFSLOPENAPI instance.
     * If `apiInstance` is null, it will attempt to initialize a new CMOFSLOPENAPI instance
     * using settings from {@link AppConfig}.
     * 
     * @param apiInstance A pre-configured instance of {@link CMOFSLOPENAPI}, or null to use default initialization.
     * @throws RuntimeException if default initialization fails due to configuration issues from AppConfig.
     */
    public ApiService(CMOFSLOPENAPI apiInstance) {
        if (apiInstance != null) {
            this.MofslApi = apiInstance;
            // System.out.println("ApiService initialized with provided CMOFSLOPENAPI instance (mock or custom).");
        } else {
            // Default behavior: initialize with AppConfig
            try {
                // System.out.println("ApiService initializing with AppConfig based CMOFSLOPENAPI.");
                this.MofslApi = new CMOFSLOPENAPI(
                        AppConfig.getApiKey(),
                        AppConfig.getBaseUrl(),
                        AppConfig.getSourceId(),
                        AppConfig.getAppBrowserName(),
                        AppConfig.getAppBrowserVersion()
                );
            } catch (Exception e) {
                System.err.println("Error initializing ApiService with AppConfig: " + e.getMessage());
                e.printStackTrace(System.err);
                throw new RuntimeException("Failed to initialize ApiService due to configuration issues.", e);
            }
        }
    }
    
    /**
     * Default constructor for normal application use.
     * Initializes the MOFSLOPENAPI client using settings from {@link AppConfig}.
     * @throws RuntimeException if initialization fails due to configuration issues from AppConfig.
     */
    public ApiService() {
        this(null); // Calls the constructor that initializes with AppConfig
    }

    /**
     * Attempts to log in to the MOFSL API using credentials from {@link AppConfig}.
     * Handles responses that may require OTP verification. Stores the authentication
     * token and client ID upon successful login.
     *
     * @return A {@link JSONObject} containing the API response. This will include:
     *         - "Status": 0 for success, 1 for failure, 2 for OTP required (based on placeholder).
     *         - "Message": A descriptive message from the API.
     *         - "AuthToken": The authentication token if login is successful (Status 0).
     *         - "ClientID": The client ID if login is successful.
     *         If an error occurs during processing, "Status" will be 1 and "Message" will contain error details.
     *         Returns an empty JSONObject with error status if the SDK returns null.
     */
    public JSONObject login() {
        JSONObject loginResponse = null;
        try {
            String userID = AppConfig.getUserId(); // Fetches from AppConfig
            String password = AppConfig.getPassword();
            String panOrDOB = AppConfig.getPanOrDOB();
            String vendorId = AppConfig.getVendorId();
            String totp = AppConfig.getTotp(); // Can be empty

            System.out.println("Attempting login for UserID: " + userID);
            loginResponse = MofslApi.Login(userID, password, panOrDOB, vendorId, totp);
            System.out.println("Login API Response: " + (loginResponse != null ? loginResponse.toString() : "null"));


            if (loginResponse != null) {
                // Based on sampleMOFSLOPENAPI.java, Status 0 is success, 2 might mean OTP
                // And AuthToken is present on success.
                // The placeholder CMOFSLOPENAPI.Login simulates Status 2 if totp is empty.
                if (loginResponse.has("Status") && loginResponse.getInt("Status") == 2) {
                    System.out.println("OTP required. Message: " + loginResponse.optString("Message"));
                    // In a real CLI, you would prompt for OTP here.
                    // For now, we just indicate it and let the caller handle it.
                    // The actual verifyotp call will be separate.
                    this.authToken = null; // Ensure no auth token if OTP is pending (though placeholder might still send one)
                } else if (loginResponse.has("AuthToken") && !loginResponse.getString("AuthToken").isEmpty()) {
                    this.authToken = loginResponse.getString("AuthToken");
                    this.clientCodeForSession = loginResponse.optString("ClientID"); // Store ClientID from response
                    System.out.println("Login successful. AuthToken and ClientID stored.");
                    if (this.clientCodeForSession.isEmpty()) {
                         System.out.println("Warning: ClientID not found or empty in login response.");
                    } else {
                        System.out.println("ClientID for this session: " + this.clientCodeForSession);
                    }
                } else {
                    // Login failed or AuthToken is missing in a supposedly successful response
                    this.authToken = null;
                    System.err.println("Login failed or AuthToken not found in response. Server Message: " + loginResponse.optString("Message"));
                }
            } else {
                 // API response was null
                this.authToken = null;
                 System.err.println("Login API call returned a null response object.");
            }
        } catch (org.json.JSONException e) { // Catching specific JSONException from the placeholder
            this.authToken = null;
            System.err.println("Error processing JSON response during login: " + e.getMessage());
            e.printStackTrace(System.err);
            loginResponse = new JSONObject(); 
            try {
                loginResponse.put("Status", 1); // Indicate failure
                loginResponse.put("Message", "Error processing JSON response during login: " + e.getMessage());
            } catch (org.json.JSONException je) { /* This should ideally not happen when putting into a new JSONObject */ }
        } catch (Exception e) { // Catch any other unexpected exceptions
            this.authToken = null;
            System.err.println("An unexpected error occurred during login: " + e.getMessage());
            e.printStackTrace(System.err);
            loginResponse = new JSONObject(); 
            try {
                loginResponse.put("Status", 1); // Indicate failure
                loginResponse.put("Message", "Unexpected error during login: " + e.getMessage());
            } catch (org.json.JSONException je) { /* This should ideally not happen */ }
        }
        // Ensure a non-null JSONObject is always returned, populated with error info if necessary.
        return loginResponse == null ? new JSONObject().put("Status", 1).put("Message", "Login returned an unexpected null response.") : loginResponse;
    }
    
    /**
     * Verifies the Time-based One-Time Password (TOTP) with the MOFSL API.
     * This is typically called after a login attempt indicates OTP is required.
     * Stores the authentication token and client ID upon successful OTP verification.
     *
     * @param otp The 6-digit OTP string provided by the user.
     * @return A {@link JSONObject} containing the API response. Expected fields are similar to `login()`:
     *         - "Status": 0 for success, 1 for failure.
     *         - "Message": Descriptive message.
     *         - "AuthToken": New token if OTP verification is successful.
     *         - "ClientID": Client ID if successful.
     *         Returns an empty JSONObject with error status if the SDK returns null or an error occurs.
     */
    public JSONObject verifyOtpAndLogin(String otp) {
        JSONObject otpResponse = null;
        try {
            System.out.println("Attempting to verify OTP: " + otp);
            otpResponse = MofslApi.verifyotp(otp);
            System.out.println("Verify OTP API Response: " + (otpResponse != null ? otpResponse.toString() : "null"));

            if (otpResponse != null) {
                if (otpResponse.has("AuthToken") && !otpResponse.getString("AuthToken").isEmpty()) {
                    this.authToken = otpResponse.getString("AuthToken");
                    this.clientCodeForSession = otpResponse.optString("ClientID");
                    System.out.println("OTP Verification successful. AuthToken stored.");
                     if (this.clientCodeForSession.isEmpty()) {
                         System.out.println("Warning: ClientID not found in OTP response.");
                    } else {
                        System.out.println("ClientID for session: " + this.clientCodeForSession);
                    }
                } else {
                    this.authToken = null;
                    System.err.println("OTP Verification failed or AuthToken not found. Message: " + otpResponse.optString("Message"));
                }
            } else {
                 this.authToken = null;
                 System.err.println("Verify OTP API returned a null response.");
            }
        } catch (JSONException e) {
            this.authToken = null;
            System.err.println("Error processing OTP JSON response: " + e.getMessage());
            e.printStackTrace();
            otpResponse = new JSONObject();
            try {
                otpResponse.put("Status", 1);
                otpResponse.put("Message", "Error processing JSON response during OTP verification: " + e.getMessage());
            } catch (JSONException je) { /* This should ideally not happen */ }
        } catch (Exception e) {
            this.authToken = null;
            System.err.println("An unexpected error occurred during OTP verification: " + e.getMessage());
            e.printStackTrace(System.err);
            otpResponse = new JSONObject();
            try {
                otpResponse.put("Status", 1); // Indicate failure
                otpResponse.put("Message", "Error processing JSON response during OTP verification: " + e.getMessage());
            } catch (org.json.JSONException je) { /* This should ideally not happen */ }
        } catch (Exception e) { // Catch any other unexpected exceptions
            this.authToken = null;
            System.err.println("An unexpected error occurred during OTP verification: " + e.getMessage());
            e.printStackTrace(System.err);
            otpResponse = new JSONObject();
            try {
                otpResponse.put("Status", 1); // Indicate failure
                otpResponse.put("Message", "Unexpected error during OTP verification: " + e.getMessage());
            } catch (org.json.JSONException je) { /* This should ideally not happen */ }
        }
        // Ensure a non-null JSONObject is always returned.
        return otpResponse == null ? new JSONObject().put("Status", 1).put("Message", "OTP verification returned an unexpected null response.") : otpResponse;
    }

    /**
     * Places an order through the MOFSL API.
     * Requires the user to be logged in (i.e., `authToken` and `clientCodeForSession` must be set).
     *
     * @param exchange The exchange (e.g., "NSE", "BSE").
     * @param symbolToken The symbol token for the instrument.
     * @param buyOrSell "BUY" or "SELL".
     * @param orderType Order type (e.g., "LIMIT", "MARKET").
     * @param productType Product type (e.g., "INTRADAY", "NORMAL").
     * @param orderDuration Order duration (e.g., "DAY", "IOC").
     * @param price The price for limit orders; 0 for market orders.
     * @param triggerPrice The trigger price for stop-loss orders.
     * @param qtyInLot Quantity in lots.
     * @param disclosedQty Disclosed quantity.
     * @param amoOrder "Y" for After Market Order, "N" otherwise.
     * @param goodTillDate Good Till Date for GTD orders (format DD-MMM-YYYY).
     * @param tag An optional tag for the order.
     * @return A {@link JSONObject} containing the API response from the order placement call.
     *         If not logged in, or if client code is missing, an error JSON is returned.
     *         If an error occurs during processing, "Status" will be 1 and "Message" will contain error details.
     */
    public JSONObject placeOrder(
            String exchange, int symbolToken, String buyOrSell,
            String orderType, String productType, String orderDuration,
            float price, int triggerPrice, int qtyInLot,
            int disclosedQty, String amoOrder, String goodTillDate, String tag) {
        
        JSONObject errorResponse = new JSONObject();
        try {
            if (this.authToken == null || this.authToken.isEmpty()) {
                System.err.println("Not logged in. Please login first.");
                errorResponse.put("Status", 1); // Indicate failure
                errorResponse.put("Message", "Login required before placing order.");
                return errorResponse;
            }
            if (this.clientCodeForSession == null || this.clientCodeForSession.isEmpty()) {
                System.err.println("ClientCode not available. Login might have been incomplete.");
                errorResponse.put("Status", 1);
                errorResponse.put("Message", "ClientCode missing. Cannot place order.");
                return errorResponse;
            }

            PlaceOrderinfo orderInfo = new PlaceOrderinfo();
            orderInfo.clientcode = this.clientCodeForSession; // Use clientCode from login
            orderInfo.exchange = exchange;
            orderInfo.symboltoken = symbolToken;
            orderInfo.buyorsell = buyOrSell;
            orderInfo.ordertype = orderType;
            orderInfo.producttype = productType;
            orderInfo.orderduration = orderDuration;
            orderInfo.price = price;
            orderInfo.triggerPrice = triggerPrice;
            orderInfo.qtyinlot = qtyInLot;
            orderInfo.disclosedqty = disclosedQty;
            orderInfo.amoorder = amoOrder;
            orderInfo.goodtilldate = goodTillDate;
            orderInfo.tag = tag;

            System.out.println("Placing order for Client: " + orderInfo.clientcode + ", Symbol: " + symbolToken);
            JSONObject orderResponse = MofslApi.PlaceOrder(orderInfo);
            System.out.println("Place Order API Response: " + (orderResponse != null ? orderResponse.toString() : "null"));
            
            if (orderResponse == null) {
                System.err.println("Place Order API returned a null response.");
                errorResponse.put("Status", 1);
                errorResponse.put("Message", "Place Order API returned null.");
                return errorResponse;
            }
            return orderResponse;

        } catch (JSONException e) {
            System.err.println("Error creating JSON for placeOrder error response (JSONException): " + e.getMessage());
            e.printStackTrace(System.err);
            // Fallback error if creating the errorResponse JSONObject itself fails.
            // This case is highly unlikely if the org.json.JSONObject placeholder is working as expected.
            return new JSONObject().put("Status", 1).put("Message", "Internal error: Failed to create JSON for error response during order placement.");
        } catch (Exception e) { // Catch any other unexpected exceptions
            System.err.println("An unexpected error occurred during placeOrder: " + e.getMessage());
            e.printStackTrace(System.err);
            try {
                errorResponse.put("Status", 1); // Indicate failure
                errorResponse.put("Message", "Unexpected error during place order: " + e.getMessage());
            } catch (org.json.JSONException je) { /* This should ideally not happen */ }
            return errorResponse;
        }
    }
    
    /**
     * Checks if the user is currently logged in (i.e., has a valid auth token).
     * @return true if logged in, false otherwise.
     */
    public boolean isLoggedIn() {
        return this.authToken != null && !this.authToken.isEmpty();
    }

    // Wrapper methods for other API functionalities follow a similar pattern:
    // 1. Check login status.
    // 2. Call the corresponding MofslApi method.
    // 3. Return the response, handling nulls from the API.

    /**
     * Fetches the order book for the currently logged-in client.
     * @return JSONObject API response, or an error JSON if not logged in or API returns null.
     */
    public JSONObject getOrderBook() {
        if (!isLoggedIn()) {
            System.err.println("ERROR: Not logged in. Please login first to get order book.");
            return new JSONObject().put("Status", 1).put("Message", "Login required to fetch order book.");
        }
        System.out.println("Fetching order book for client: " + this.clientCodeForSession);
        JSONObject response = MofslApi.GetOrderBook(this.clientCodeForSession);
        return response == null ? new JSONObject().put("Status", 1).put("Message", "GetOrderBook returned null response.") : response;
    }

    /**
     * Fetches the trade book for the currently logged-in client.
     * @return JSONObject API response, or an error JSON if not logged in or API returns null.
     */
    public JSONObject getTradeBook() {
        if (!isLoggedIn()) {
            System.err.println("ERROR: Not logged in. Please login first to get trade book.");
            return new JSONObject().put("Status", 1).put("Message", "Login required to fetch trade book.");
        }
        System.out.println("Fetching trade book for client: " + this.clientCodeForSession);
        JSONObject response = MofslApi.GetTradeBook(this.clientCodeForSession);
        return response == null ? new JSONObject().put("Status", 1).put("Message", "GetTradeBook returned null response.") : response;
    }
    
    /**
     * Fetches the net positions for the currently logged-in client.
     * @return JSONObject API response, or an error JSON if not logged in or API returns null.
     */
    public JSONObject getNetPosition() {
        if (!isLoggedIn()) {
             System.err.println("ERROR: Not logged in. Please login first to get net positions.");
            return new JSONObject().put("Status", 1).put("Message", "Login required to fetch net positions.");
        }
        System.out.println("Fetching net positions for client: " + this.clientCodeForSession);
        JSONObject response = MofslApi.GetNetPosition(this.clientCodeForSession);
        return response == null ? new JSONObject().put("Status", 1).put("Message", "GetNetPosition returned null response.") : response;
    }

    /**
     * Fetches the available funds for the currently logged-in client.
     * @return JSONObject API response, or an error JSON if not logged in or API returns null.
     */
    public JSONObject getFunds() {
        if (!isLoggedIn()) {
            System.err.println("ERROR: Not logged in. Please login first to get funds.");
            return new JSONObject().put("Status", 1).put("Message", "Login required to fetch funds.");
        }
        System.out.println("Fetching funds for client: " + this.clientCodeForSession);
        JSONObject response = MofslApi.GetFunds(this.clientCodeForSession);
        return response == null ? new JSONObject().put("Status", 1).put("Message", "GetFunds returned null response.") : response;
    }
    
    /**
     * Fetches the RMS (Risk Management System) limits for the currently logged-in client.
     * @return JSONObject API response, or an error JSON if not logged in or API returns null.
     */
    public JSONObject getRMSLimit() {
        if (!isLoggedIn()) {
            System.err.println("ERROR: Not logged in. Please login first to get RMS limits.");
            return new JSONObject().put("Status", 1).put("Message", "Login required to fetch RMS limits.");
        }
        System.out.println("Fetching RMS limits for client: " + this.clientCodeForSession);
        JSONObject response = MofslApi.GetRMSLimit(this.clientCodeForSession);
        return response == null ? new JSONObject().put("Status", 1).put("Message", "GetRMSLimit returned null response.") : response;
    }
}
