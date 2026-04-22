import GlobalModel from '../model/Global.js';
import { ParsedModelFinder } from '../util/model.js';
import ExtensionAPI from '../core/ExtensionAPI.js';
import { DimensionDefinitionLoose, ScaleDataValue, DimensionName } from '../util/types.js';
import Axis from './Axis.js';
import { BoundingRect } from '../util/graphic.js';
import { MatrixArray } from 'zrender/lib/core/matrix.js';
import ComponentModel from '../model/Component.js';
import { RectLike } from 'zrender/lib/core/BoundingRect.js';
import type { PrepareCustomInfo } from '../chart/custom/CustomSeries.js';
export interface CoordinateSystemCreator {
    create: (ecModel: GlobalModel, api: ExtensionAPI) => CoordinateSystemMaster[];
    dimensions?: DimensionName[];
    getDimensionsInfo?: () => DimensionDefinitionLoose[];
}
/**
 * The instance get from `CoordinateSystemManger` is `CoordinateSystemMaster`.
 */
export interface CoordinateSystemMaster {
    dimensions: DimensionName[];
    model?: ComponentModel;
    update?: (ecModel: GlobalModel, api: ExtensionAPI) => void;
    convertToPixel?(ecModel: GlobalModel, finder: ParsedModelFinder, value: ScaleDataValue | ScaleDataValue[]): number | number[];
    convertFromPixel?(ecModel: GlobalModel, finder: ParsedModelFinder, pixelValue: number | number[]): number | number[];
    containPoint(point: number[]): boolean;
    getAxes?: () => Axis[];
    axisPointerEnabled?: boolean;
    getTooltipAxes?: (dim: DimensionName | 'auto') => {
        baseAxes: Axis[];
        otherAxes: Axis[];
    };
    /**
     * Get layout rect or coordinate system
     */
    getRect?: () => RectLike;
}
/**
 * For example: cartesian is CoordinateSystem.
 * series.coordinateSystem is CoordinateSystem.
 */
export interface CoordinateSystem {
    type: string;
    /**
     * Master of coordinate system. For example:
     * Grid is master of cartesian.
     */
    master?: CoordinateSystemMaster;
    dimensions: DimensionName[];
    model?: ComponentModel;
    /**
     * @param data
     * @param reserved Defined by the coordinate system itself
     * @param out
     * @return {Array.<number>} point Point in global pixel coordinate system.
     */
    dataToPoint(data: ScaleDataValue | ScaleDataValue[], reserved?: any, out?: number[]): number[];
    /**
     * Some coord sys (like Parallel) might do not have `pointToData`,
     * or the meaning of this kind of features is not clear yet.
     * @param point point Point in global pixel coordinate system.
     * @param clamp Clamp range
     * @return data
     */
    pointToData?(point: number[], clamp?: boolean): number | number[];
    containPoint(point: number[]): boolean;
    getAxes?: () => Axis[];
    getAxis?: (dim?: DimensionName) => Axis;
    getBaseAxis?: () => Axis;
    getOtherAxis?: (baseAxis: Axis) => Axis;
    clampData?: (data: ScaleDataValue[], out?: number[]) => number[];
    getRoamTransform?: () => MatrixArray;
    getArea?: (tolerance?: number) => CoordinateSystemClipArea;
    getBoundingRect?: () => BoundingRect;
    getAxesByScale?: (scaleType: string) => Axis[];
    prepareCustoms?: PrepareCustomInfo;
}
/**
 * Like GridModel, PolarModel, ...
 */
export interface CoordinateSystemHostModel extends ComponentModel {
    coordinateSystem?: CoordinateSystemMaster;
}
/**
 * Clip area will be returned by getArea of CoordinateSystem.
 * It is used to clip the graphic elements with the contain methods.
 */
export interface CoordinateSystemClipArea {
    contain(x: number, y: number): boolean;
}
export declare function isCoordinateSystemType<T extends CoordinateSystem, S = T['type']>(coordSys: CoordinateSystem, type: S): coordSys is T;
