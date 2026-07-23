declare class Browser {
    firefox: boolean;
    ie: boolean;
    edge: boolean;
    newEdge: boolean;
    weChat: boolean;
    version: string | number;
}
declare class Env {
    browser: Browser;
    node: boolean;
    wxa: boolean;
    worker: boolean;
    svgSupported: boolean;
    touchEventsSupported: boolean;
    pointerEventsSupported: boolean;
    domSupported: boolean;
    transformSupported: boolean;
    transform3dSupported: boolean;
    hasGlobalWindow: boolean;
}
declare const env: Env;
export default env;
