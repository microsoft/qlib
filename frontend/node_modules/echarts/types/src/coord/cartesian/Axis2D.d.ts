import Axis from '../Axis.js';
import { DimensionName, OrdinalSortInfo } from '../../util/types.js';
import Scale from '../../scale/Scale.js';
import CartesianAxisModel, { CartesianAxisPosition } from './AxisModel.js';
import Grid from './Grid.js';
import { OptionAxisType } from '../axisCommonTypes.js';
interface Axis2D {
    /**
     * Transform global coord to local coord,
     * i.e. let localCoord = axis.toLocalCoord(80);
     */
    toLocalCoord(coord: number): number;
    /**
     * Transform global coord to local coord,
     * i.e. let globalCoord = axis.toLocalCoord(40);
     */
    toGlobalCoord(coord: number): number;
}
declare class Axis2D extends Axis {
    /**
     * Axis position
     *  - 'top'
     *  - 'bottom'
     *  - 'left'
     *  - 'right'
     */
    readonly position: CartesianAxisPosition;
    /**
     * Index of axis, can be used as key
     * Injected outside.
     */
    index: number;
    /**
     * Axis model. Injected outside
     */
    model: CartesianAxisModel;
    /**
     * Injected outside.
     */
    grid: Grid;
    constructor(dim: DimensionName, scale: Scale, coordExtent: [number, number], axisType?: OptionAxisType, position?: CartesianAxisPosition);
    /**
     * Implemented in <module:echarts/coord/cartesian/Grid>.
     * @return If not on zero of other axis, return null/undefined.
     *         If no axes, return an empty array.
     */
    getAxesOnZeroOf: () => Axis2D[];
    isHorizontal(): boolean;
    /**
     * Each item cooresponds to this.getExtent(), which
     * means globalExtent[0] may greater than globalExtent[1],
     * unless `asc` is input.
     *
     * @param {boolean} [asc]
     * @return {Array.<number>}
     */
    getGlobalExtent(asc?: boolean): [number, number];
    pointToData(point: number[], clamp?: boolean): number;
    /**
     * Set ordinalSortInfo
     * @param info new OrdinalSortInfo
     */
    setCategorySortInfo(info: OrdinalSortInfo): boolean;
}
export default Axis2D;
