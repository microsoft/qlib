package MOFSLOPENAPI.Client;

/**
 * Placeholder class for `MOFSLOPENAPI.Client.PlaceOrderinfo`.
 * This class mimics the structure of the `PlaceOrderinfo` object expected by the
 * placeholder `CMOFSLOPENAPI.PlaceOrder` method. It contains public fields
 * for order parameters.
 *
 * In a real integration, this class would be provided by the actual `MOFSLOPENAPI_V3.1.jar`.
 * This placeholder should be removed when the real SDK is used and properly configured.
 */
public class PlaceOrderinfo {
    // These fields are based on common parameters for placing an order.
    // The actual fields and their types should match the real SDK's PlaceOrderinfo class.
    
    public String clientcode;
    public String exchange;
    public int symboltoken;
    public String buyorsell;
    public String ordertype;
    public String producttype;
    public String orderduration;
    public float price;
    public int triggerPrice;
    public int qtyinlot;
    public int disclosedqty;
    public String amoorder;
    public String goodtilldate;
    public String tag;

    public PlaceOrderinfo() {
        // Initialize default values if necessary
    }
}
