import Element from 'zrender/lib/Element.js';
interface ControllerHost {
    target: Element;
    zoom?: number;
    zoomLimit?: {
        min?: number;
        max?: number;
    };
}
/**
 * For geo and graph.
 */
export declare function updateViewOnPan(controllerHost: ControllerHost, dx: number, dy: number): void;
/**
 * For geo and graph.
 */
export declare function updateViewOnZoom(controllerHost: ControllerHost, zoomDelta: number, zoomX: number, zoomY: number): void;
export {};
