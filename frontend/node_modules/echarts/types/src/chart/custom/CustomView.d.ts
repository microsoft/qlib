import ChartView from '../../view/Chart.js';
import { EventQueryItem, ECActionEvent, Payload, StageHandlerProgressParams } from '../../util/types.js';
import Element from 'zrender/lib/Element.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import CustomSeriesModel from './CustomSeries.js';
export default class CustomChartView extends ChartView {
    static type: string;
    readonly type: string;
    private _data;
    private _progressiveEls;
    render(customSeries: CustomSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    incrementalPrepareRender(customSeries: CustomSeriesModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    incrementalRender(params: StageHandlerProgressParams, customSeries: CustomSeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    eachRendered(cb: (el: Element) => boolean | void): void;
    filterForExposedEvent(eventType: string, query: EventQueryItem, targetEl: Element, packedEvent: ECActionEvent): boolean;
}
