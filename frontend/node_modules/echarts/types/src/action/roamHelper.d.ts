import type View from '../coord/View.js';
import type ExtensionAPI from '../core/ExtensionAPI.js';
import type { Payload } from '../util/types.js';
export interface RoamPayload extends Payload {
    dx: number;
    dy: number;
    zoom: number;
    originX: number;
    originY: number;
}
export declare function updateCenterAndZoom(view: View, payload: RoamPayload, zoomLimit?: {
    min?: number;
    max?: number;
}, api?: ExtensionAPI): {
    center: number[];
    zoom: number;
};
