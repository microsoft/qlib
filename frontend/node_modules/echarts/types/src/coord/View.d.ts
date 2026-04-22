import * as matrix from 'zrender/lib/core/matrix.js';
import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import Transformable from 'zrender/lib/core/Transformable.js';
import { CoordinateSystemMaster, CoordinateSystem } from './CoordinateSystem.js';
import GlobalModel from '../model/Global.js';
import { ParsedModelFinder } from '../util/model.js';
import type ExtensionAPI from '../core/ExtensionAPI.js';
export declare type ViewCoordSysTransformInfoPart = Pick<Transformable, 'x' | 'y' | 'scaleX' | 'scaleY'>;
declare class View extends Transformable implements CoordinateSystemMaster, CoordinateSystem {
    readonly type: string;
    static dimensions: string[];
    readonly dimensions: string[];
    readonly name: string;
    zoomLimit: {
        max?: number;
        min?: number;
    };
    /**
     * Represents the transform brought by roam/zoom.
     * If `View['_viewRect']` applies roam transform,
     * we can get the final displayed rect.
     */
    private _roamTransformable;
    /**
     * Represents the transform from `View['_rect']` to `View['_viewRect']`.
     */
    protected _rawTransformable: Transformable;
    private _rawTransform;
    /**
     * This is a user specified point on the source, which will be
     * located to the center of the `View['_viewRect']`.
     * The unit this the same as `View['_rect']`.
     */
    private _center;
    private _zoom;
    /**
     * The rect of the source, where the measure is used by "data" and "center".
     * Has nothing to do with roam/zoom.
     * The unit is defined by the source. For example,
     * for geo source the unit is lat/lng,
     * for SVG source the unit is the same as the width/height defined in SVG.
     */
    private _rect;
    /**
     * The visible rect on the canvas. Has nothing to do with roam/zoom.
     * The unit of `View['_viewRect']` is pixel of the canvas.
     */
    private _viewRect;
    constructor(name?: string);
    setBoundingRect(x: number, y: number, width: number, height: number): BoundingRect;
    /**
     * @return {module:zrender/core/BoundingRect}
     */
    getBoundingRect(): BoundingRect;
    setViewRect(x: number, y: number, width: number, height: number): void;
    /**
     * Transformed to particular position and size
     */
    protected _transformTo(x: number, y: number, width: number, height: number): void;
    /**
     * Set center of view
     */
    setCenter(centerCoord: (number | string)[], api: ExtensionAPI): void;
    setZoom(zoom: number): void;
    /**
     * Get default center without roam
     */
    getDefaultCenter(): number[];
    getCenter(): number[];
    getZoom(): number;
    getRoamTransform(): matrix.MatrixArray;
    /**
     * Remove roam
     */
    private _updateCenterAndZoom;
    /**
     * Update transform props on `this` based on the current
     * `this._roamTransformable` and `this._rawTransformable`.
     */
    protected _updateTransform(): void;
    getTransformInfo(): {
        roam: ViewCoordSysTransformInfoPart;
        raw: ViewCoordSysTransformInfoPart;
    };
    getViewRect(): BoundingRect;
    /**
     * Get view rect after roam transform
     */
    getViewRectAfterRoam(): BoundingRect;
    /**
     * Convert a single (lon, lat) data item to (x, y) point.
     */
    dataToPoint(data: number[], noRoam?: boolean, out?: number[]): number[];
    /**
     * Convert a (x, y) point to (lon, lat) data
     */
    pointToData(point: number[]): number[];
    convertToPixel(ecModel: GlobalModel, finder: ParsedModelFinder, value: number[]): number[];
    convertFromPixel(ecModel: GlobalModel, finder: ParsedModelFinder, pixel: number[]): number[];
    /**
     * @implements
     */
    containPoint(point: number[]): boolean;
}
export default View;
