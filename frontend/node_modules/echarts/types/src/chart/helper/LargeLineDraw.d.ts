import * as graphic from '../../util/graphic.js';
import SeriesData from '../../data/SeriesData.js';
import { StageHandlerProgressParams, LineStyleOption } from '../../util/types.js';
import Model from '../../model/Model.js';
import Element from 'zrender/lib/Element.js';
interface LargeLinesCommonOption {
    polyline?: boolean;
    lineStyle?: LineStyleOption & {
        curveness?: number;
    };
}
/**
 * Data which can support large lines.
 */
declare type LargeLinesData = SeriesData<Model<LargeLinesCommonOption> & {
    seriesIndex?: number;
}>;
declare class LargeLineDraw {
    group: graphic.Group;
    private _newAdded;
    /**
     * Update symbols draw by new data
     */
    updateData(data: LargeLinesData): void;
    /**
     * @override
     */
    incrementalPrepareUpdate(data: LargeLinesData): void;
    /**
     * @override
     */
    incrementalUpdate(taskParams: StageHandlerProgressParams, data: LargeLinesData): void;
    /**
     * @override
     */
    remove(): void;
    eachRendered(cb: (el: Element) => boolean | void): void;
    private _create;
    private _setCommon;
    private _clear;
}
export default LargeLineDraw;
