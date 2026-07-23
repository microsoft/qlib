import env from './core/env.js';
var dpr = 1;
if (env.hasGlobalWindow) {
    dpr = Math.max(window.devicePixelRatio
        || (window.screen && window.screen.deviceXDPI / window.screen.logicalXDPI)
        || 1, 1);
}
export var debugMode = 0;
export var devicePixelRatio = dpr;
export var DARK_MODE_THRESHOLD = 0.4;
export var DARK_LABEL_COLOR = '#333';
export var LIGHT_LABEL_COLOR = '#ccc';
export var LIGHTER_LABEL_COLOR = '#eee';
