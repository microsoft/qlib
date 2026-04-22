import VisualMapView from './VisualMapView.js';
import ContinuousModel from './ContinuousModel.js';
import GlobalModel from '../../model/Global.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
declare class ContinuousView extends VisualMapView {
    static type: string;
    type: string;
    visualMapModel: ContinuousModel;
    private _shapes;
    private _dataInterval;
    private _handleEnds;
    private _orient;
    private _useHandle;
    private _hoverLinkDataIndices;
    private _dragging;
    private _hovering;
    private _firstShowIndicator;
    init(ecModel: GlobalModel, api: ExtensionAPI): void;
    doRender(visualMapModel: ContinuousModel, ecModel: GlobalModel, api: ExtensionAPI, payload: {
        type: string;
        from: string;
    }): void;
    private _buildView;
    private _renderEndsText;
    private _renderBar;
    private _createHandle;
    private _createIndicator;
    private _dragHandle;
    private _resetInterval;
    /**
     * @private
     * @param {(number|string)} handleIndex 0 or 1 or 'all'
     * @param {number} dx
     * @param {number} dy
     */
    private _updateInterval;
    private _updateView;
    private _createBarVisual;
    private _makeColorGradient;
    private _createBarPoints;
    private _createBarGroup;
    private _updateHandle;
    private _showIndicator;
    private _enableHoverLinkToSeries;
    private _enableHoverLinkFromSeries;
    private _doHoverLinkToSeries;
    private _hoverLinkFromSeriesMouseOver;
    private _hideIndicator;
    private _clearHoverLinkToSeries;
    private _clearHoverLinkFromSeries;
    private _applyTransform;
    private _dispatchHighDown;
    /**
     * @override
     */
    dispose(): void;
}
export default ContinuousView;
