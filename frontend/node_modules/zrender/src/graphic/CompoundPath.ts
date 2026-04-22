// CompoundPath to improve performance

import Path from './Path';
import PathProxy from '../core/PathProxy';

export interface CompoundPathShape {
    paths: Path[]
}

export default class CompoundPath extends Path {

    type = 'compound'

    shape: CompoundPathShape

    private _updatePathDirty() {
        const paths = this.shape.paths;
        let dirtyPath = this.shapeChanged();
        for (let i = 0; i < paths.length; i++) {
            // Mark as dirty if any subpath is dirty
            dirtyPath = dirtyPath || paths[i].shapeChanged();
        }
        if (dirtyPath) {
            this.dirtyShape();
        }
    }

    beforeBrush() {
        this._updatePathDirty();
        const paths = this.shape.paths || [];
        const scale = this.getGlobalScale();
        // Update path scale
        for (let i = 0; i < paths.length; i++) {
            if (!paths[i].path) {
                paths[i].createPathProxy();
            }
            paths[i].path.setScale(scale[0], scale[1], paths[i].segmentIgnoreThreshold);
        }
    }

    buildPath(ctx: PathProxy | CanvasRenderingContext2D, shape: CompoundPathShape) {
        const paths = shape.paths || [];
        for (let i = 0; i < paths.length; i++) {
            paths[i].buildPath(ctx, paths[i].shape, true);
        }
    }

    afterBrush() {
        const paths = this.shape.paths || [];
        for (let i = 0; i < paths.length; i++) {
            paths[i].pathUpdated();
        }
    }

    getBoundingRect() {
        this._updatePathDirty.call(this);
        return Path.prototype.getBoundingRect.call(this);
    }
}