declare const wx: {
    getSystemInfoSync: Function
};

class Browser {
    firefox = false
    ie = false
    edge = false
    newEdge = false
    weChat = false
    version: string | number
}

class Env {
    browser = new Browser()
    node = false
    wxa = false
    worker = false

    svgSupported = false
    touchEventsSupported = false
    pointerEventsSupported = false
    domSupported = false
    transformSupported = false
    transform3dSupported = false

    hasGlobalWindow = typeof window !== 'undefined'
}

const env = new Env();

if (typeof wx === 'object' && typeof wx.getSystemInfoSync === 'function') {
    env.wxa = true;
    env.touchEventsSupported = true;
}
else if (typeof document === 'undefined' && typeof self !== 'undefined') {
    // In worker
    env.worker = true;
}
else if (!env.hasGlobalWindow || 'Deno' in window) {
    // In node
    env.node = true;
    env.svgSupported = true;
}
else {
    detect(navigator.userAgent, env);
}

// Zepto.js
// (c) 2010-2013 Thomas Fuchs
// Zepto.js may be freely distributed under the MIT license.

function detect(ua: string, env: Env) {
    const browser = env.browser;
    const firefox = ua.match(/Firefox\/([\d.]+)/);
    const ie = ua.match(/MSIE\s([\d.]+)/)
        // IE 11 Trident/7.0; rv:11.0
        || ua.match(/Trident\/.+?rv:(([\d.]+))/);
    const edge = ua.match(/Edge?\/([\d.]+)/); // IE 12 and 12+

    const weChat = (/micromessenger/i).test(ua);

    if (firefox) {
        browser.firefox = true;
        browser.version = firefox[1];
    }
    if (ie) {
        browser.ie = true;
        browser.version = ie[1];
    }

    if (edge) {
        browser.edge = true;
        browser.version = edge[1];
        browser.newEdge = +edge[1].split('.')[0] > 18;
    }

    // It is difficult to detect WeChat in Win Phone precisely, because ua can
    // not be set on win phone. So we do not consider Win Phone.
    if (weChat) {
        browser.weChat = true;
    }

    env.svgSupported = typeof SVGRect !== 'undefined';
    env.touchEventsSupported = 'ontouchstart' in window && !browser.ie && !browser.edge;
    env.pointerEventsSupported = 'onpointerdown' in window
        && (browser.edge || (browser.ie && +browser.version >= 11));
    env.domSupported = typeof document !== 'undefined';

    const style = document.documentElement.style;

    env.transform3dSupported = (
        // IE9 only supports transform 2D
        // transform 3D supported since IE10
        // we detect it by whether 'transition' is in style
        (browser.ie && 'transition' in style)
        // edge
        || browser.edge
        // webkit
        || (('WebKitCSSMatrix' in window) && ('m11' in new WebKitCSSMatrix()))
        // gecko-based browsers
        || 'MozPerspective' in style
    ) // Opera supports CSS transforms after version 12
      && !('OTransition' in style);

    // except IE 6-8 and very old firefox 2-3 & opera 10.1
    // other browsers all support `transform`
    env.transformSupported = env.transform3dSupported
        // transform 2D is supported in IE9
        || (browser.ie && +browser.version >= 9);

}


export default env;
