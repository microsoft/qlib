import Model from '../../model/Model.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { AxisPointerOption } from './AxisPointerModel.js';
import Axis from '../../coord/Axis.js';
import SeriesModel from '../../model/Series.js';
import { CommonAxisPointerOption, Dictionary } from '../../util/types.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
import ComponentModel from '../../model/Component.js';
import { CoordinateSystemMaster } from '../../coord/CoordinateSystem.js';
interface LinkGroup {
    mapper: AxisPointerOption['link'][number]['mapper'];
    /**
     * { [axisKey]: AxisInfo }
     */
    axesInfo: Dictionary<AxisInfo>;
}
interface AxisInfo {
    axis: Axis;
    key: string;
    coordSys: CoordinateSystemMaster;
    axisPointerModel: Model<CommonAxisPointerOption>;
    triggerTooltip: boolean;
    triggerEmphasis: boolean;
    involveSeries: boolean;
    snap: boolean;
    useHandle: boolean;
    seriesModels: SeriesModel[];
    linkGroup?: LinkGroup;
    seriesDataCount?: number;
}
interface CollectionResult {
    /**
     * { [coordSysKey]: { [axisKey]: AxisInfo } }
     */
    coordSysAxesInfo: Dictionary<Dictionary<AxisInfo>>;
    /**
     * { [axisKey]: AxisInfo }
     */
    axesInfo: Dictionary<AxisInfo>;
    /**
     * { [coordSysKey]: { CoordinateSystemMaster } }
     */
    coordSysMap: Dictionary<CoordinateSystemMaster>;
    seriesInvolved: boolean;
}
export declare function collect(ecModel: GlobalModel, api: ExtensionAPI): CollectionResult;
export declare function fixValue(axisModel: AxisBaseModel): void;
export declare function getAxisInfo(axisModel: AxisBaseModel): AxisInfo;
export declare function getAxisPointerModel(axisModel: AxisBaseModel): Model<CommonAxisPointerOption>;
/**
 * @param {module:echarts/model/Model} model
 * @return {string} unique key
 */
export declare function makeKey(model: ComponentModel): string;
export {};
