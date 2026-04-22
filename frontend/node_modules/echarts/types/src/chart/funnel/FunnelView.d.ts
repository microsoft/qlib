import ChartView from '../../view/Chart.js';
import FunnelSeriesModel from './FunnelSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
declare class FunnelView extends ChartView {
    static type: "funnel";
    type: "funnel";
    private _data;
    ignoreLabelLineUpdate: boolean;
    render(seriesModel: FunnelSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    remove(): void;
    dispose(): void;
}
export default FunnelView;
