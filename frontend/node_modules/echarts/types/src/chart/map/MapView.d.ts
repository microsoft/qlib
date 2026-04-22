import ChartView from '../../view/Chart.js';
import MapSeries from './MapSeries.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Payload } from '../../util/types.js';
declare class MapView extends ChartView {
    static type: "map";
    readonly type: "map";
    private _mapDraw;
    render(mapModel: MapSeries, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    remove(): void;
    dispose(): void;
    private _renderSymbols;
}
export default MapView;
