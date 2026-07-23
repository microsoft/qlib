import ChartView from '../../view/Chart.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import EffectScatterSeriesModel from './EffectScatterSeries.js';
declare class EffectScatterView extends ChartView {
    static readonly type = "effectScatter";
    readonly type = "effectScatter";
    private _symbolDraw;
    init(): void;
    render(seriesModel: EffectScatterSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    _getClipShape(seriesModel: EffectScatterSeriesModel): import("../../coord/CoordinateSystem").CoordinateSystemClipArea;
    updateTransform(seriesModel: EffectScatterSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    _updateGroupTransform(seriesModel: EffectScatterSeriesModel): void;
    remove(ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default EffectScatterView;
