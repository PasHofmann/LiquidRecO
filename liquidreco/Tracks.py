import numpy as np
from liquidreco import parameters

class SimpleTrack:
    def __init__(self, track_id):
        self.track_id = track_id
        self.line_ids = np.array([], dtype = np.int64)
        self.lines = []

    def add_line(self, line):
        self.line_ids = np.append(self.line_ids, line.line_id)
        self.lines.append(line)
        line.track_id = self.track_id

    def __len__(self):
        return len(self.lines)


    def Track_generator(self, lines, i):
            if lines[i].track_id == -1:
                self.add_line(lines[i])

                if lines[i].neighbour_id_in != -1:
                    self.Track_generator(lines, lines[i].neighbour_id_in)
                if lines[i].neighbour_id_f != -1:
                    self.Track_generator(lines, lines[i].neighbour_id_f)
            return
    def show(self):
        print("\n Track_id = ", self.track_id, "      line_ids = ", self.line_ids)
        return

    def show_lines(self):
        for line in self.lines:
            line.show()

    def get_line(self, line_id):
        matches = np.where(self.line_ids == line_id)[0]
        if len(matches) == 0:
            raise ValueError(f"line_id {line_id} is not in track {self.track_id}")
        return self.lines[matches[0]]

    def line_split(self, line_index, point, in_and_f):
        line = self.get_line(line_index)
        old_p_f = line.p_f.copy()
        old_neighbour_id_f = line.neighbour_id_f

        new_line_id = len(in_and_f)
        in_and_f = np.append(in_and_f,[[point, old_p_f]], axis=0)

        line.p_f[:] = point.copy()
        in_and_f[line.line_id, 1, :] = point

        line.neighbour_id_f = new_line_id
        new_line = Line(in_and_f, new_line_id)
        new_line.neighbour_id_in = line.line_id
        new_line.neighbour_id_f = old_neighbour_id_f

        if old_neighbour_id_f != -1:
            f_neighbour = self.get_line(old_neighbour_id_f)
            if f_neighbour.neighbour_id_in == line.line_id:
                f_neighbour.neighbour_id_in = new_line_id
            elif f_neighbour.neighbour_id_f == line.line_id:
                f_neighbour.neighbour_id_f = new_line_id
            else:
                print("The line was not the neighbour of its neighbour?")

        self.add_line(new_line)
        return in_and_f

    def z_range(self, plane):
        coordinate = {"zy": 0, "xz": 1}.get(plane)
        if coordinate is None:
            raise ValueError("Not an acceptable plane: " + plane)
        values = [line.p_in[coordinate] for line in self.lines] + [line.p_f[coordinate] for line in self.lines]
        return min(values), max(values)
    


class Line:
    def __init__(self, in_and_f, line_id, total_charge = 0.0):
        self._set(in_and_f[line_id, 0, :], in_and_f[line_id, 1, :], line_id, total_charge)
        """
        self.line_id = line_id
        self.p_in = in_and_f[line_id, 0, :].copy()
        self.p_f = in_and_f[line_id, 1, :].copy()
        self.total_charge = total_charge
        self.track_id = -1
        self.neighbour_id_in = -1
        self.neighbour_id_f = -1
        """
    @classmethod
    def from_points(cls, p_in, p_f, line_id, total_charge = 0.0):
        line = cls.__new__(cls)
        line._set(p_in, p_f, line_id, total_charge)
        return line

    def _set(self, p_in, p_f, line_id, total_charge):
        self.line_id = line_id
        self.p_in = np.array(p_in, dtype = float)
        self.p_f = np.array(p_f, dtype = float)
        self.total_charge = total_charge
        self.track_id = -1
        self.neighbour_id_in = -1
        self.neighbour_id_f = -1
        

    def show(self):
        print("\n line_id = ", self.line_id, "\n  p_in = ", self.p_in, "\n  p_f = ", self.p_f, "\n  track_id = ", self.track_id, "\n    neighbour_id_in = ", self.neighbour_id_in, "\n  neighbour_id_f = ", self.neighbour_id_f)
        return

    def length(self):
        return np.linalg.norm(self.p_f-self.p_in)

    def get_end_point(self, end_point):
        if isinstance(end_point, str):
            if end_point == "in":
                return self.p_in
            elif end_point == "f":
                return self.p_f
        elif np.array_equal(end_point, self.p_in):
            return "in"
        elif np.array_equal(end_point, self.p_f):
            return "f"
        raise ValueError(f"{end_point} is not a valid end_point")

    def get_other_end_point(self, end_point):
        if isinstance(end_point, str):
            if end_point == "in":
                return self.p_f
            elif end_point == "f":
                return self.p_in
        elif np.linalg.norm(end_point - self.p_in)<1e-6:
            return self.p_f
        elif np.linalg.norm(end_point - self.p_f) <1e-6:
            return self.p_in

        raise ValueError(f"{end_point} is not a valid end_point")

    def get_point_on_line(self, t):
        dir = self.p_f - self.p_in
        return self.p_in + t * dir


class VertexCandidate:
    def __init__(self, position, vertex_id, line):
        self.vertex_id = vertex_id
        self.position = position
        self.connected_line_ids = [line.line_id]
        self.neighbour_vertices = []
        

    def add_neighbour_line(self, line):
        
        if np.linalg.norm(line.p_in-self.position) >1e-1 and np.linalg.norm(line.p_f-self.position)>1e-1:
            print("\n These line is not a neighbour")
            return
        if line.line_id in self.connected_line_ids:
            #print("\n Line is already in neighbour list")
            return
        self.connected_line_ids.append(line.line_id)

    def add_neighbour_vertex(self, neighbour_vertex):
        if not (set(self.connected_line_ids) & set(neighbour_vertex.connected_line_ids)):
            print("\n Vertices are not neighbours")
            return
        if self.vertex_id in neighbour_vertex.neighbour_vertices:
            print("\n Already neighbours")
            return
        if self.vertex_id == neighbour_vertex.vertex_id:
            print("\n Vertex cannot be its own neighbour")
            return
        self.neighbour_vertices.append(neighbour_vertex.vertex_id)
        neighbour_vertex.neighbour_vertices.append(self.vertex_id)




class SoftVertexCandidate:
    def __init__(self, position, vertex_id, charge = 0.):
        self.vertex_id = vertex_id
        self.position = position
        self.attachments = []
        self.neighbour_vertices = []
        self.mut_exc_vertices = []
        self.charge = charge

    def connected_line_ids(self):
        return {a[0] for a in self.attachments}

    def add_neighbour_line(self, line, end, gap):
        if any(a[0] == line.line_id and a[1] == end for a in self.attachments):
            return
        self.attachments.append((line.line_id, end, gap))

    def get_connection_endpoint_of_line(self, line_id):
        for i, (lid, end, gap) in enumerate(self.attachments):
            if lid == line_id:
                return end
        raise ValueError(f"Line {line_id} is not attached to vertex {self.vertex_id}")


            

#TODO
    def add_neighbour_vertex(self, neighbour_vertex):
        if not (set(self.connected_line_ids()) & set(neighbour_vertex.connected_line_ids())):
            print("\n Vertices are not neighbours")
            return
        if self.vertex_id in neighbour_vertex.neighbour_vertices:
            print("\n Already neighbours")
            return
        if self.vertex_id == neighbour_vertex.vertex_id:
            print("\n Vertex cannot be its own neighbour")
            return
        self.neighbour_vertices.append(neighbour_vertex.vertex_id)
        neighbour_vertex.neighbour_vertices.append(self.vertex_id)

    

        #self.neighbour_id_in = -1
        #self.neighbour_id_f = -1 


class SoftTrack:
    def __init__(self, track_id, line, plane):
        self.plane = plane
        self.track_id = track_id
        self.line_ids = np.array([line.line_id], dtype = np.int64)
        self.lines = [line]
        self.gaps = []
        self.attachment_point = (line.line_id, "both")
        self.total_charge = line.total_charge
        self.vertices = []
        self.nodes = []

    def track_length(self):
        total_length = 0
        for line in self.lines:
            total_length += line.length()
        return total_length

    def charge_density(self):
        z_min, z_max = self.z_range()
        return self.total_charge / (z_max-z_min)

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def add_line(self, line, gap, free_end):
        self.line_ids = np.append(self.line_ids, line.line_id)
        self.lines.append(line)
        self.gaps.append(gap)
        self.total_charge += line.total_charge
        if free_end == "in":
            self.attachment_point = (line.line_id, "in")
        elif free_end == "f":
            self.attachment_point = (line.line_id, "f")
        else:
            raise ValueError(f"next_attachment_point {free_end} is not in valid")

    def copy(self, new_id = None):
        new = SoftTrack.__new__(SoftTrack)
        new.plane = self.plane
        new.track_id = new_id
        new.line_ids = self.line_ids.copy()
        new.lines = list(self.lines)
        new.gaps = list(self.gaps)
        new.attachment_point = self.attachment_point
        new.total_charge = self.total_charge
        new.vertices = list(self.vertices)
        new.nodes = list(self.nodes)
        return new

    def __len__(self):
        return len(self.lines)

    def show(self):
        print("\n Track_id = ", self.track_id, "      line_ids = ", self.line_ids)
        return

    def show_lines(self):
        for line in self.lines:
            line.show()

    def get_line(self, line_id):
        matches = np.where(self.line_ids == line_id)[0]
        if len(matches) == 0:
            raise ValueError(f"line_id {line_id} is not in track {self.track_id}")
        return self.lines[matches[0]]


    def z_range(self, fiber_dist = parameters.z_range_fiber_dist):
        coordinate = {"zy": 0, "xz": 1}.get(self.plane)
        if coordinate is None:
            raise ValueError("Not an acceptable plane: " + self.plane)
        values = [line.p_in[coordinate] for line in self.lines] + [line.p_f[coordinate] for line in self.lines]
        lo, hi = min(values), max(values)
        if hi-lo < fiber_dist:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - fiber_dist, mid + fiber_dist
        return lo, hi


    def connect_lines_in_track(self):
        for vertex in self.vertices:
            for attachment in vertex.attachments:
                if attachment[0] in self.line_ids:
                    line = self.get_line(attachment[0])
                    if attachment[1] == "in":
                        line.p_in = vertex.position
                    elif attachment[1] == "f":
                        line.p_f = vertex.position
                    else:
                        raise ValueError(f"{attachment[1]} is not an acceptable input for attachment[1]")

    def z_coordinate(self):
        if self.plane == "zy":
            return 0
        elif self.plane == "xz":
            return 1
        else:
            raise ValueError(f"{self.plane} is not a valid plane")

    def not_z_coordinate(self):
        if self.plane == "zy":
            return 1
        elif self.plane == "xz":
            return 0
        else:
            raise ValueError(f"{self.plane} is not a valid plane")

    def monotonous(self):
        z_coordinate = self.z_coordinate()
        turning_points = []
        signs = []
        if self.__len__() == 1:
            line = self.lines[0]
            signs = [sign_func(line.p_f[z_coordinate]- line.p_in[z_coordinate])]
            return (True, signs, turning_points)
        first_vertex = self.vertices[0]
        attached_line_end = first_vertex.get_connection_endpoint_of_line(self.line_ids[0])
        line = self.lines[0]
        start_point = line.get_other_end_point(attached_line_end)
        sign = sign_func(line.get_other_end_point(start_point)[z_coordinate]-start_point[z_coordinate])
        if sign !=0:
            signs.append(sign)
        for i in range(self.__len__()-1):
            line = self.lines[i+1]
            vertex =self.vertices[i]
            attached_line_end = vertex.get_connection_endpoint_of_line(self.line_ids[i+1])
            p1 = line.get_end_point(attached_line_end)
            p2 = line.get_other_end_point(p1)
            current_sign = sign_func(p2[z_coordinate]-p1[z_coordinate])
            if current_sign == 0:
                continue
            if sign == 0:
                sign = current_sign
                signs.append(sign)
            if sign == current_sign:
                continue
            else:
                sign = current_sign
                signs.append(sign)
                turning_points.append(vertex)
        for i, turning_point in enumerate(turning_points):
            if signs[i]<0:
                extrema_type = "min"
            else:
                extrema_type = "max"
            turning_points[i] = turning_point_class(turning_point, extrema_type, self.plane, i,) 
        if len(turning_points) == 0:
            return (True, signs, turning_points)
        else:
            return (False, signs, turning_points)

    def get_point_sequence(self):
        #z_coordinate = self.z_coordinate()
        points = []
        if self.__len__() == 1:
            line = self.lines[0]
            points = [line.p_in, line.p_f]
            return points
        first_vertex = self.vertices[0]
        attached_line_end = first_vertex.get_connection_endpoint_of_line(self.line_ids[0])
        line = self.lines[0]
        p1 = line.get_other_end_point(attached_line_end)
        points.append(p1)
        for line in self.lines:
            p2 = line.get_other_end_point(p1)
            points.append(p2)
            p1 = p2
        return points

    def get_point_in_function_of_z(self, z_value, tol = parameters.z_range_fiber_dist):
        z_index = self.z_coordinate()
        not_z_index = self.not_z_coordinate()
        points = []
        for line in self.lines:
             # order the endpoints by z: p_low has the smaller z, p_high the larger
            p_low, p_high = sorted((line.p_in, line.p_f), key=lambda p: p[z_index])
            direction = p_high-p_low
            if direction[z_index] < 1e-9:
                if abs(p_low[z_index]-z_value) < tol:
                    points.append(line.p_in)
                    points.append(line.p_f)
                continue
            if p_low[z_index]-tol <= z_value <= p_high[z_index]+tol:
                a = direction[not_z_index]/direction[z_index]
                b = p_low[not_z_index] - a*p_low[z_index]
                point = [0,0]
                point[z_index] = z_value
                point[not_z_index] = a*z_value + b
                points.append(np.array(point))
        return points


    def get_vertex_index(self, point, tol = 1e-9):
        for i, vertex in enumerate(self.vertices):
            if np.linalg.norm(point-vertex.position)<tol:
                return i
        raise ValueError(f"point = {point} is not a vertex")

    def get_endpoints_of_track(self):
        if len(self.lines)==1:
            return (self.lines[0].p_in, self.lines[0].p_f)
        first_line = self.lines[0]
        last_line = self.lines[-1]
        first_vertex = self.vertices[0]
        last_vertex = self.vertices[-1]
        first_end = first_vertex.get_connection_endpoint_of_line(first_line.line_id)
        p1 = first_line.get_other_end_point(first_end)
        last_end = last_vertex.get_connection_endpoint_of_line(last_line.line_id)
        p2 = last_line.get_other_end_point(last_end)
        return (p1, p2)

    def slice_tracks(self, p1, p2, tol = 1e-9):
            if np.linalg.norm(p1-p2)<tol:
                return None
            subtrack = self.copy()
            (p_in, p_f) = self.get_endpoints_of_track()
            last_line_index = self.__len__()

            if np.linalg.norm(p1-p_in)<tol:
                i1 = 0
            elif np.linalg.norm(p1-p_f)<tol:
                i1 = last_line_index
            else:
                i1 = self.get_vertex_index(p1)+1
            if np.linalg.norm(p2-p_in)<tol:
                i2 = 0
            elif np.linalg.norm(p2-p_f)<tol:
                i2 = last_line_index
            else:
                i2 = self.get_vertex_index(p2)+1
            a = min(i1, i2)
            b = max(i1, i2)
            p_end = p1 if i1 > i2 else p2
            
            subtrack.line_ids = subtrack.line_ids[a:b]
            subtrack.lines = subtrack.lines[a:b]
            subtrack.gaps = subtrack.gaps[a:b-1]
            subtrack.total_charge = sum(line.total_charge for line in subtrack.lines)
            subtrack.vertices = subtrack.vertices[a:b-1]
            subtrack.attachment_point = (subtrack.line_ids[-1], subtrack.lines[-1].get_end_point(p_end))
            return subtrack
               
            

    def split_into_monotonous_subtracks(self):
        monotonous, signs, turning_points = self.monotonous()
        if monotonous:
            return signs, turning_points, [self.copy()]
        else:
            subtracks = []
            first_point, last_point = self.get_endpoints_of_track()
            points = [first_point]
            for turning_point in turning_points:
                points.append(turning_point.position)
            points.append(last_point)
            for i in range(len(points)-1):
                subtracks.append(self.slice_tracks(points[i], points[i+1]))
            return signs, turning_points, subtracks

    def reverse_track(self):
        reversed_track = self.copy()
        reversed_track.nodes = self.nodes[::-1]
        if len(self.lines) == 1:
            line = self.lines[0]
            flipped = Line.from_points(line.p_f, line.p_in, line.line_id, line.total_charge)
            reversed_track.lines = [flipped]
            reversed_track.line_ids = self.line_ids.copy()
            reversed_track.attachment_point = (line.line_id, "both")
            return reversed_track
        reversed_track.line_ids = reversed_track.line_ids[::-1].copy()
        reversed_track.lines = reversed_track.lines[::-1]
        reversed_track.gaps = reversed_track.gaps[::-1]
        reversed_track.vertices = reversed_track.vertices[::-1]
        reversed_track.attachment_point = (reversed_track.line_ids[-1],reversed_track.lines[-1].get_end_point(self.get_endpoints_of_track()[0])) 
        return reversed_track   


def sign_func(number, threshold=1e-9):
    if abs(number) < threshold:
        return 0
    else:
        return np.sign(number)


class ThreeDTrack:
    def __init__(self, track_id, line):
        self.track_id = track_id
        self.line_ids = np.array([line.line_id], dtype = np.int64)
        self.lines = [line]
        self.points = [line.p_in, line.p_f]

    def add_point_to_track(self, point, free_line_id):
        self.line_ids = np.append(self.line_ids, free_line_id)
        p_in = self.lines[-1].p_f
        line = Line.from_points(p_in, point, free_line_id)
        self.lines.append(line)
        self.points.append(point)

class turning_point_class(SoftVertexCandidate):
    def __init__(self, vertex, extrema_type, plane, id):
        super().__init__(vertex.position, vertex.vertex_id)
        self.attachments = list(vertex.attachments)
        self.neighbour_vertices = list(vertex.neighbour_vertices)
        self.mut_exc_vertices = list(vertex.mut_exc_vertices)
        self.extrema_type = extrema_type
        self.plane = plane
        self.id = id

    def get_z_value(self):
        if self.plane == "zy":
            return self.position[0]
        elif self.plane == "xz":
            return self.position[1]
        else:
            raise ValueError(f"{self.plane} is not an recognized plane")
        
class turning_point_matches:
    def __init__(self, matched_list):
        self.matched_list = matched_list
        self.indices_zy = set()
        self.indices_xz = set()
        for entry in matched_list:
            if isinstance(entry, (list, tuple)):
                if entry[0].plane != "zy":
                    raise ValueError(f"{entry[0].plane} is not of the plane zy")
                self.indices_zy.add(entry[0].id)
                if entry[1].plane != "xz":
                    raise ValueError(f"{entry[0].plane} is not of the plane xz")
                self.indices_xz.add(entry[1].id)
            elif isinstance(entry, turning_point_class):
                if entry.plane == "zy":
                    self.indices_zy.add(entry.id)
                elif entry.plane == "xz":
                    self.indices_xz.add(entry.id)
                else:
                    raise ValueError(f"{entry.plane} is not a known plane")
                
    def add_match(self, match):
        if isinstance(match, (list, tuple)):
            if match[0].plane != "zy":
                raise ValueError(f"{match[0].plane} is not of the plane zy")
            if match[1].plane != "xz":
                raise ValueError(f"{match[0].plane} is not of the plane xz")
            if match[0].id in self.indices_zy:
                raise ValueError(f" {match[0].id} is already paired in zy")
            if match[1].id in self.indices_xz:
                raise ValueError(f"{match[1].id} is already paired in xz")
            self.matched_list.append(match)
            self.indices_zy.add(match[0].id)
            self.indices_xz.add(match[1].id)
        if isinstance(match, turning_point_class):
            if match.plane == "zy":
                self.matched_list.append(match)
                self.indices_zy.add(match.id)
            elif match.plane == "xz":
                self.matched_list.append(match)
                self.indices_xz.add(match.id)
            else:
                raise ValueError(f"{match.plane} is not a known plane")
        return self

    def copy(self):
        new = turning_point_matches(list(self.matched_list))
        return new

    def get_number_of_pairs(self):
        i = 0
        for entry in self.matched_list:
            if isinstance(entry, (list, tuple)):
                i+=1
        return i

    def get_pair_list(self):
        pair_list = []
        for entry in self.matched_list:
            if isinstance(entry, (list, tuple)):
                pair_list.append(entry)
        return pair_list

    def get_match_cost(self, n_zy, n_xz, sigma_z = parameters.turning_point_matches_sigma_z, unmatched_penalty = parameters.turning_point_matches_unmatched_penalty):
        cost = 0.0
        pair_list = self.get_pair_list()
        for turningpoint_zy, turningpoint_xz in pair_list:
            dz = turningpoint_zy.get_z_value() - turningpoint_xz.get_z_value()
            cost += (dz/sigma_z)**2
        n_unmatched = n_zy+n_xz-len(pair_list)*2
        return cost + unmatched_penalty * n_unmatched

    def get_matched_point(self, point, point_plane):
        for entry in self.matched_list:
            if isinstance(entry, (list, tuple)): 
                if point_plane == "zy":
                    if np.linalg.norm(entry[0].position-point) < 1e-9:
                        return entry[1]
                elif point_plane == "xz":
                    if np.linalg.norm(entry[1].position-point) < 1e-9:
                        return entry[0]
                else:
                    raise ValueError(f"{point_plane} is not a valid plane")
        return None

class Node:
    def __init__(self, position, charge, plane):
        self.position = position
        self.charge = charge
        self.plane = plane