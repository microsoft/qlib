import { createAxisLabels, calculateCategoryInterval } from './axisTickLabelBuilder.js';
import Scale from '../scale/Scale.js';
import { DimensionName, ScaleDataValue, ScaleTick } from '../util/types.js';
import Model from '../model/Model.js';
import { AxisBaseOption, CategoryAxisBaseOption, OptionAxisType } from './axisCommonTypes.js';
import { AxisBaseModel } from './AxisBaseModel.js';
interface TickCoord {
    coord: number;
    tickValue?: ScaleTick['value'];
}
/**
 * Base class of Axis.
 */
declare class Axis {
    /**
     * Axis type
     *  - 'category'
     *  - 'value'
     *  - 'time'
     *  - 'log'
     */
    type: OptionAxisType;
    readonly dim: DimensionName;
    scale: Scale;
    private _extent;
    model: AxisBaseModel;
    onBand: CategoryAxisBaseOption['boundaryGap'];
    inverse: AxisBaseOption['inverse'];
    constructor(dim: DimensionName, scale: Scale, extent: [number, number]);
    /**
     * If axis extent contain given coord
     */
    contain(coord: number): boolean;
    /**
     * If axis extent contain given data
     */
    containData(data: ScaleDataValue): boolean;
    /**
     * Get coord extent.
     */
    getExtent(): [number, number];
    /**
     * Get precision used for formatting
     */
    getPixelPrecision(dataExtent?: [number, number]): number;
    /**
     * Set coord extent
     */
    setExtent(start: number, end: number): void;
    /**
     * Convert data to coord. Data is the rank if it has an ordinal scale
     */
    dataToCoord(data: ScaleDataValue, clamp?: boolean): number;
    /**
     * Convert coord to data. Data is the rank if it has an ordinal scale
     */
    coordToData(coord: number, clamp?: boolean): number;
    /**
     * Convert pixel point to data in axis
     */
    pointToData(point: number[], clamp?: boolean): number;
    /**
     * Different from `zrUtil.map(axis.getTicks(), axis.dataToCoord, axis)`,
     * `axis.getTicksCoords` considers `onBand`, which is used by
     * `boundaryGap:true` of category axis and splitLine and splitArea.
     * @param opt.tickModel default: axis.model.getModel('axisTick')
     * @param opt.clamp If `true`, the first and the last
     *        tick must be at the axis end points. Otherwise, clip ticks
     *        that outside the axis extent.
     */
    getTicksCoords(opt?: {
        tickModel?: Model;
        clamp?: boolean;
    }): TickCoord[];
    getMinorTicksCoords(): TickCoord[][];
    getViewLabels(): ReturnType<typeof createAxisLabels>['labels'];
    getLabelModel(): Model<AxisBaseOption['axisLabel']>;
    /**
     * Notice here we only get the default tick model. For splitLine
     * or splitArea, we should pass the splitLineModel or splitAreaModel
     * manually when calling `getTicksCoords`.
     * In GL, this method may be overridden to:
     * `axisModel.getModel('axisTick', grid3DModel.getModel('axisTick'));`
     */
    getTickModel(): Model;
    /**
     * Get width of band
     */
    getBandWidth(): number;
    /**
     * Get axis rotate, by degree.
     */
    getRotate: () => number;
    /**
     * Only be called in category axis.
     * Can be overridden, consider other axes like in 3D.
     * @return Auto interval for cateogry axis tick and label
     */
    calculateCategoryInterval(): ReturnType<typeof calculateCategoryInterval>;
}
export default Axis;
