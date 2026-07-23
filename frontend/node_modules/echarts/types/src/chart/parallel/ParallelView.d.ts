import ChartView from '../../view/Chart.js';
import ParallelSeriesModel from './ParallelSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { StageHandlerProgressParams, Payload } from '../../util/types.js';
declare class ParallelView extends ChartView {
    static type: string;
    type: string;
    private _dataGroup;
    private _data;
    private _initialized;
    private _progressiveEls;
    init(): void;
    /**
     * @override
     */
    render(seriesModel: ParallelSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    incrementalPrepareRender(seriesModel: ParallelSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalRender(taskParams: StageHandlerProgressParams, seriesModel: ParallelSeriesModel, ecModel: GlobalModel): void;
    remove(): void;
}
export default ParallelView;
