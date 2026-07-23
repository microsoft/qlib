import TooltipHTMLContent from './TooltipHTMLContent.js';
import TooltipRichContent from './TooltipRichContent.js';
import { TooltipMarker } from '../../util/format.js';
import Model from '../../model/Model.js';
import ComponentView from '../../view/Component.js';
import { CallbackDataParams, CommonTooltipOption } from '../../util/types.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import TooltipModel, { TooltipOption } from './TooltipModel.js';
import Element from 'zrender/lib/Element.js';
import { ECData } from '../../util/innerStore.js';
import { DataByCoordSys } from '../axisPointer/axisTrigger.js';
interface ShowTipPayload {
    type?: 'showTip';
    from?: string;
    tooltip?: ECData['tooltipConfig']['option'];
    dataByCoordSys?: DataByCoordSys[];
    tooltipOption?: CommonTooltipOption<TooltipCallbackDataParams | TooltipCallbackDataParams[]>;
    seriesIndex?: number;
    dataIndex?: number;
    name?: string;
    x?: number;
    y?: number;
    position?: TooltipOption['position'];
    dispatchAction?: ExtensionAPI['dispatchAction'];
}
interface HideTipPayload {
    type?: 'hideTip';
    from?: string;
    dispatchAction?: ExtensionAPI['dispatchAction'];
}
declare type TooltipCallbackDataParams = CallbackDataParams & {
    axisDim?: string;
    axisIndex?: number;
    axisType?: string;
    axisId?: string;
    axisValue?: string | number;
    axisValueLabel?: string;
    marker?: TooltipMarker;
};
declare class TooltipView extends ComponentView {
    static type: "tooltip";
    type: "tooltip";
    private _renderMode;
    private _tooltipModel;
    private _ecModel;
    private _api;
    private _tooltipContent;
    private _refreshUpdateTimeout;
    private _lastX;
    private _lastY;
    private _ticket;
    private _showTimout;
    private _lastDataByCoordSys;
    private _cbParamsList;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(tooltipModel: TooltipModel, ecModel: GlobalModel, api: ExtensionAPI): void;
    private _initGlobalListener;
    private _keepShow;
    /**
     * Show tip manually by
     * dispatchAction({
     *     type: 'showTip',
     *     x: 10,
     *     y: 10
     * });
     * Or
     * dispatchAction({
     *      type: 'showTip',
     *      seriesIndex: 0,
     *      dataIndex or dataIndexInside or name
     * });
     *
     *  TODO Batch
     */
    manuallyShowTip(tooltipModel: TooltipModel, ecModel: GlobalModel, api: ExtensionAPI, payload: ShowTipPayload): void;
    manuallyHideTip(tooltipModel: TooltipModel, ecModel: GlobalModel, api: ExtensionAPI, payload: HideTipPayload): void;
    private _manuallyAxisShowTip;
    private _tryShow;
    private _showOrMove;
    private _showAxisTooltip;
    private _showSeriesItemTooltip;
    private _showComponentItemTooltip;
    private _showTooltipContent;
    private _getNearestPoint;
    _updatePosition(tooltipModel: Model<TooltipOption>, positionExpr: TooltipOption['position'], x: number, // Mouse x
    y: number, // Mouse y
    content: TooltipHTMLContent | TooltipRichContent, params: TooltipCallbackDataParams | TooltipCallbackDataParams[], el?: Element): void;
    private _updateContentNotChangedOnAxis;
    private _hide;
    dispose(ecModel: GlobalModel, api: ExtensionAPI): void;
}
export default TooltipView;
