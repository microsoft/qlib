package org.json;

/**
 * Placeholder class for org.json.JSONObject.
 * This is a minimal implementation to allow the application to compile
 * and demonstrate functionality without the actual org.json library.
 * Replace this with the actual org.json library if available and
 * properly configured in the project's dependencies.
 */
public class JSONObject {
    private java.util.Map<String, Object> map;

    public JSONObject() {
        this.map = new java.util.HashMap<>();
    }

    public JSONObject(String jsonString) {
        // This is a very basic placeholder, not a real JSON parser
        this.map = new java.util.HashMap<>();
        // Example: "{\"key\":\"value\"}"
        if (jsonString != null && jsonString.startsWith("{") && jsonString.endsWith("}")) {
            String[] pairs = jsonString.substring(1, jsonString.length() - 1).split(",");
            for (String pair : pairs) {
                String[] keyValue = pair.split(":");
                if (keyValue.length == 2) {
                    String key = keyValue[0].trim().replace("\"", "");
                    String value = keyValue[1].trim().replace("\"", "");
                    this.map.put(key, value);
                }
            }
        }
    }

    public String getString(String key) {
        Object value = map.get(key);
        return value == null ? "" : value.toString();
    }

    public boolean has(String key) {
        return map.containsKey(key);
    }
    
    public Object get(String key) {
        return map.get(key);
    }

    public JSONObject put(String key, Object value) {
        map.put(key, value);
        return this;
    }
    
    public JSONObject put(String key, boolean value) {
        map.put(key, value);
        return this;
    }

    public JSONObject put(String key, int value) {
        map.put(key, value);
        return this;
    }

    public JSONObject put(String key, long value) {
        map.put(key, value);
        return this;
    }
    
    public JSONObject put(String key, double value) {
        map.put(key, value);
        return this;
    }
    
    public int getInt(String key) {
        Object value = map.get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        try {
            return Integer.parseInt(getString(key));
        } catch (NumberFormatException e) {
            return 0; // Or throw an exception
        }
    }


    @Override
    public String toString() {
        // Basic toString, not a full JSON serializer
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (java.util.Map.Entry<String, Object> entry : map.entrySet()) {
            if (!first) {
                sb.append(",");
            }
            sb.append("\"").append(entry.getKey()).append("\":");
            if (entry.getValue() instanceof String) {
                sb.append("\"").append(entry.getValue()).append("\"");
            } else {
                sb.append(entry.getValue());
            }
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }
}
