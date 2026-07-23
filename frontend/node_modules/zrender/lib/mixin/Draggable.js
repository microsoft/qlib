var Param = (function () {
    function Param(target, e) {
        this.target = target;
        this.topTarget = e && e.topTarget;
    }
    return Param;
}());
var Draggable = (function () {
    function Draggable(handler) {
        this.handler = handler;
        handler.on('mousedown', this._dragStart, this);
        handler.on('mousemove', this._drag, this);
        handler.on('mouseup', this._dragEnd, this);
    }
    Draggable.prototype._dragStart = function (e) {
        var draggingTarget = e.target;
        while (draggingTarget && !draggingTarget.draggable) {
            draggingTarget = draggingTarget.parent || draggingTarget.__hostTarget;
        }
        if (draggingTarget) {
            this._draggingTarget = draggingTarget;
            draggingTarget.dragging = true;
            this._x = e.offsetX;
            this._y = e.offsetY;
            this.handler.dispatchToElement(new Param(draggingTarget, e), 'dragstart', e.event);
        }
    };
    Draggable.prototype._drag = function (e) {
        var draggingTarget = this._draggingTarget;
        if (draggingTarget) {
            var x = e.offsetX;
            var y = e.offsetY;
            var dx = x - this._x;
            var dy = y - this._y;
            this._x = x;
            this._y = y;
            draggingTarget.drift(dx, dy, e);
            this.handler.dispatchToElement(new Param(draggingTarget, e), 'drag', e.event);
            var dropTarget = this.handler.findHover(x, y, draggingTarget).target;
            var lastDropTarget = this._dropTarget;
            this._dropTarget = dropTarget;
            if (draggingTarget !== dropTarget) {
                if (lastDropTarget && dropTarget !== lastDropTarget) {
                    this.handler.dispatchToElement(new Param(lastDropTarget, e), 'dragleave', e.event);
                }
                if (dropTarget && dropTarget !== lastDropTarget) {
                    this.handler.dispatchToElement(new Param(dropTarget, e), 'dragenter', e.event);
                }
            }
        }
    };
    Draggable.prototype._dragEnd = function (e) {
        var draggingTarget = this._draggingTarget;
        if (draggingTarget) {
            draggingTarget.dragging = false;
        }
        this.handler.dispatchToElement(new Param(draggingTarget, e), 'dragend', e.event);
        if (this._dropTarget) {
            this.handler.dispatchToElement(new Param(this._dropTarget, e), 'drop', e.event);
        }
        this._draggingTarget = null;
        this._dropTarget = null;
    };
    return Draggable;
}());
export default Draggable;
