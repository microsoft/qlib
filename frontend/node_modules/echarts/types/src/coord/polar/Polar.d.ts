import RadiusAxis from './RadiusAxis.js';
import AngleAxis from './AngleAxis.js';
import PolarModel from './PolarModel.js';
import { CoordinateSystem, CoordinateSystemMaster, CoordinateSystemClipArea } from '../CoordinateSystem.js';
import GlobalModel from '../../model/Global.js';
import { ParsedModelFinder } from '../../util/model.js';
import { ScaleDataValue } from '../../util/types.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
export declare const polarDimensions: string[];
interface Polar {
    update(ecModel: GlobalModel, api: ExtensionAPI): void;
}
declare class Polar implements CoordinateSystem, CoordinateSystemMaster {
    readonly name: string;
    readonly dimensions: string[];
    readonly type = "polar";
    /**
     * x of polar center
     */
    cx: number;
    /**
     * y of polar center
     */
    cy: number;
    private _radiusAxis;
    private _angleAxis;
    axisPointerEnabled: boolean;
    model: PolarModel;
    constructor(name: string);
    /**
     * If contain coord
     */
    containPoint(point: number[]): boolean;
    /**
     * If contain data
     */
    containData(data: number[]): boolean;
    getAxis(dim: 'radius' | 'angle'): AngleAxis | RadiusAxis;
    getAxes(): (AngleAxis | RadiusAxis)[];
    /**
     * Get axes by type of scale
     */
    getAxesByScale(scaleType: 'ordinal' | 'interval' | 'time' | 'log'): (AngleAxis | RadiusAxis)[];
    getAngleAxis(): AngleAxis;
    getRadiusAxis(): RadiusAxis;
    getOtherAxis(axis: AngleAxis | RadiusAxis): AngleAxis | RadiusAxis;
    /**
     * Base axis will be used on stacking.
     *
     */
    getBaseAxis(): AngleAxis | RadiusAxis;
    getTooltipAxes(dim: 'radius' | 'angle' | 'auto'): {
        baseAxes: (AngleAxis | RadiusAxis)[];
        otherAxes: (AngleAxis | RadiusAxis)[];
    };
    /**
     * Convert a single data item to (x, y) point.
     * Parameter data is an array which the first element is radius and the second is angle
     */
    dataToPoint(data: ScaleDataValue[], clamp?: boolean): number[];
    /**
     * Convert a (x, y) point to data
     */
    pointToData(point: number[], clamp?: boolean): number[];
    /**
     * Convert a (x, y) point to (radius, angle) coord
     */
    pointToCoord(point: number[]): number[];
    /**
     * Convert a (radius, angle) coord to (x, y) point
     */
    coordToPoint(coord: number[]): number[];
    /**
     * Get ring area of cartesian.
     * Area will have a contain function to determine if a point is in the coordinate system.
     */
    getArea(): PolarArea;
    convertToPixel(ecModel: GlobalModel, finder: ParsedModelFinder, value: ScaleDataValue[]): number[];
    convertFromPixel(ecModel: GlobalModel, finder: ParsedModelFinder, pixel: number[]): number[];
}
interface PolarArea extends CoordinateSystemClipArea {
    cx: number;
    cy: number;
    r0: number;
    r: number;
    startAngle: number;
    endAngle: number;
    clockwise: boolean;
}
export default Polar;
