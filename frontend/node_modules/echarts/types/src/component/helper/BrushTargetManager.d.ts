import * as graphic from '../../util/graphic.js';
import { BrushPanelConfig, BrushControllerEvents, BrushType, BrushAreaRange, BrushDimensionMinMax } from './BrushController.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import GridModel from '../../coord/cartesian/GridModel.js';
import GeoModel from '../../coord/geo/GeoModel.js';
import { CoordinateSystemMaster } from '../../coord/CoordinateSystem.js';
import Cartesian2D from '../../coord/cartesian/Cartesian2D.js';
import Geo from '../../coord/geo/Geo.js';
import GlobalModel from '../../model/Global.js';
import { BrushAreaParam, BrushAreaParamInternal } from '../brush/BrushModel.js';
import SeriesModel from '../../model/Series.js';
import { ModelFinderObject } from '../../util/model.js';
declare type COORD_CONVERTS_INDEX = 0 | 1;
declare type BrushableCoordinateSystem = Cartesian2D | Geo;
declare type BrushTargetBuilderKey = 'grid' | 'geo';
/**
 * There can be multiple axes in a single targetInfo. Consider the case
 * of `grid` component, a targetInfo represents a grid which contains one or more
 * cartesian and one or more axes. And consider the case of parallel system,
 * which has multiple axes in a coordinate system.
 */
interface BrushTargetInfo {
    panelId: string;
    coordSysModel: CoordinateSystemMaster['model'];
    coordSys: BrushableCoordinateSystem;
    coordSyses: BrushableCoordinateSystem[];
    getPanelRect: GetPanelRect;
}
export interface BrushTargetInfoCartesian2D extends BrushTargetInfo {
    gridModel: GridModel;
    coordSys: Cartesian2D;
    coordSyses: Cartesian2D[];
    xAxisDeclared: boolean;
    yAxisDeclared: boolean;
}
export interface BrushTargetInfoGeo extends BrushTargetInfo {
    geoModel: GeoModel;
    coordSysModel: GeoModel;
    coordSys: Geo;
    coordSyses: Geo[];
}
declare type GetPanelRect = () => graphic.BoundingRect;
declare class BrushTargetManager {
    private _targetInfoList;
    /**
     * @param finder contains Index/Id/Name of xAxis/yAxis/geo/grid
     *        Each can be {number|Array.<number>}. like: {xAxisIndex: [3, 4]}
     * @param opt.include include coordinate system types.
     */
    constructor(finder: ModelFinderObject, ecModel: GlobalModel, opt?: {
        include?: BrushTargetBuilderKey[];
    });
    setOutputRanges(areas: BrushControllerEvents['brush']['areas'], ecModel: GlobalModel): BrushAreaParam[];
    matchOutputRanges<T extends (Parameters<BrushTargetManager['findTargetInfo']>[0] & {
        brushType: BrushType;
        range: BrushAreaRange;
    })>(areas: T[], ecModel: GlobalModel, cb: (area: T, coordRange: ReturnType<ConvertCoord>['values'], coordSys: BrushableCoordinateSystem, ecModel: GlobalModel) => void): void;
    /**
     * the `areas` is `BrushModel.areas`.
     * Called in layout stage.
     * convert `area.coordRange` to global range and set panelId to `area.range`.
     */
    setInputRanges(areas: BrushAreaParamInternal[], ecModel: GlobalModel): void;
    makePanelOpts(api: ExtensionAPI, getDefaultBrushType?: (targetInfo: BrushTargetInfo) => BrushType): BrushPanelConfig[];
    controlSeries(area: BrushAreaParamInternal, seriesModel: SeriesModel, ecModel: GlobalModel): boolean;
    /**
     * If return Object, a coord found.
     * If return true, global found.
     * Otherwise nothing found.
     */
    findTargetInfo(area: ModelFinderObject & {
        panelId?: string;
    }, ecModel: GlobalModel): BrushTargetInfo | true;
}
declare type ConvertCoord = (to: COORD_CONVERTS_INDEX, coordSys: BrushableCoordinateSystem, rangeOrCoordRange: BrushAreaRange, clamp?: boolean) => {
    values: BrushAreaRange;
    xyMinMax: BrushDimensionMinMax[];
};
export default BrushTargetManager;
