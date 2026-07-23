import * as clazzUtil from '../util/clazz.js';
import SeriesModel from '../model/Series.js';
import GlobalModel from '../model/Global.js';
import ExtensionAPI from '../core/ExtensionAPI.js';
import Element from 'zrender/lib/Element.js';
import { Payload, ViewRootGroup, ECActionEvent, EventQueryItem, StageHandlerProgressParams, ECElementEvent } from '../util/types.js';
import { SeriesTask } from '../core/Scheduler.js';
interface ChartView {
    /**
     * Rendering preparation in progressive mode.
     * Implement it if needed.
     */
    incrementalPrepareRender(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    /**
     * Render in progressive mode.
     * Implement it if needed.
     * @param params See taskParams in `stream/task.js`
     */
    incrementalRender(params: StageHandlerProgressParams, seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    /**
     * Update transform directly.
     * Implement it if needed.
     */
    updateTransform(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void | {
        update: true;
    };
    /**
     * The view contains the given point.
     * Implement it if needed.
     */
    containPoint(point: number[], seriesModel: SeriesModel): boolean;
    /**
     * Pass only when return `true`.
     * Implement it if needed.
     */
    filterForExposedEvent(eventType: string, query: EventQueryItem, targetEl: Element, packedEvent: ECActionEvent | ECElementEvent): boolean;
}
declare class ChartView {
    type: string;
    readonly group: ViewRootGroup;
    readonly uid: string;
    readonly renderTask: SeriesTask;
    /**
     * Ignore label line update in global stage. Will handle it in chart itself.
     * Used in pie / funnel
     */
    ignoreLabelLineUpdate: boolean;
    __alive: boolean;
    __model: SeriesModel;
    __id: string;
    static protoInitialize: void;
    constructor();
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    /**
     * Highlight series or specified data item.
     */
    highlight(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    /**
     * Downplay series or specified data item.
     */
    downplay(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    /**
     * Remove self.
     */
    remove(ecModel: GlobalModel, api: ExtensionAPI): void;
    /**
     * Dispose self.
     */
    dispose(ecModel: GlobalModel, api: ExtensionAPI): void;
    updateView(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    updateLayout(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    updateVisual(seriesModel: SeriesModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload): void;
    /**
     * Traverse the new rendered elements.
     *
     * It will traverse the new added element in progressive rendering.
     * And traverse all in normal rendering.
     */
    eachRendered(cb: (el: Element) => boolean | void): void;
    static markUpdateMethod(payload: Payload, methodName: keyof ChartView): void;
    static registerClass: clazzUtil.ClassManager['registerClass'];
}
export declare type ChartViewConstructor = typeof ChartView & clazzUtil.ExtendableConstructor & clazzUtil.ClassManager;
export default ChartView;
