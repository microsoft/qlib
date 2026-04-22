import { PointLike } from './Point';
import BoundingRect from './BoundingRect';
import { MatrixArray } from './matrix';
declare class OrientedBoundingRect {
    private _corners;
    private _axes;
    private _origin;
    constructor(rect?: BoundingRect, transform?: MatrixArray);
    fromBoundingRect(rect: BoundingRect, transform?: MatrixArray): void;
    intersect(other: OrientedBoundingRect, mtv?: PointLike): boolean;
    private _intersectCheckOneSide;
    private _getProjMinMaxOnAxis;
}
export default OrientedBoundingRect;
