import DataZoomView from './DataZoomView.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Payload } from '../../util/types.js';
import SliderZoomModel from './SliderZoomModel.js';
declare class SliderZoomView extends DataZoomView {
    static type: string;
    type: string;
    dataZoomModel: SliderZoomModel;
    private _displayables;
    private _orient;
    private _range;
    /**
     * [coord of the first handle, coord of the second handle]
     */
    private _handleEnds;
    /**
     * [length, thick]
     */
    private _size;
    private _handleWidth;
    private _handleHeight;
    private _location;
    private _brushStart;
    private _brushStartTime;
    private _dragging;
    private _brushing;
    private _dataShadowInfo;
    private _shadowData;
    private _shadowDim;
    private _shadowSize;
    private _shadowPolygonPts;
    private _shadowPolylinePts;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    render(dataZoomModel: SliderZoomModel, ecModel: GlobalModel, api: ExtensionAPI, payload: Payload & {
        from: string;
        type: string;
    }): void;
    dispose(): void;
    private _clear;
    private _buildView;
    private _resetLocation;
    private _positionGroup;
    private _getViewExtent;
    private _renderBackground;
    private _renderDataShadow;
    private _prepareDataShadowInfo;
    private _renderHandle;
    private _resetInterval;
    private _updateInterval;
    private _updateView;
    private _updateDataInfo;
    private _formatLabel;
    /**
     * @param isEmphasis true: show, false: hide
     */
    private _showDataInfo;
    private _onDragMove;
    private _onDragEnd;
    private _onClickPanel;
    private _onBrushStart;
    private _onBrushEnd;
    private _onBrush;
    private _updateBrushRect;
    /**
     * This action will be throttled.
     */
    _dispatchZoomAction(realtime: boolean): void;
    private _findCoordRect;
}
export default SliderZoomView;
