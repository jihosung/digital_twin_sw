"""
 * @file        custom_gnss.py
 * @brief       Custom GNSS class for GNSS data processing and analysis.
 * 
 * @authors     Jaehwan Lee (idljh5529@gmail.com)      
 *
 * @date        2025-08-11 Released by AI Lab, Hanyang University
 * 
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import folium

# ROS bag 읽기 라이브러리 설정
use_rosbag = False
use_rosbags = False

try:
    import rosbag
    use_rosbag = True
    print("rosbag 라이브러리를 사용합니다.")
except ImportError:
    try:
        from pathlib import Path
        from rosbags.highlevel import AnyReader
        use_rosbags = True
        print("rosbags.highlevel.AnyReader를 사용합니다.")
    except ImportError:
        print("rosbag과 rosbags 라이브러리 모두 사용할 수 없습니다. 시뮬레이션 데이터를 사용합니다.")

class GNSSProcessor:
    """GNSS 데이터 처리를 위한 클래스"""
    
    def __init__(self):
        # WGS84 상수
        self.WGS84_A = 6378137.0  # Semi-major axis (meters)
        self.WGS84_B = 6356752.314245  # Semi-minor axis (meters)
        
    def parse_nmea_file(self, file_path: str) -> Dict:
        """
        NMEA 로그 파일을 파싱하여 GGA와 RMC 데이터를 추출합니다.
        
        Args:
            file_path (str): NMEA 파일 경로
            
        Returns:
            Dict: 파싱된 NMEA 데이터
        """
        gga_data = []
        rmc_data = []
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                        gga_data.append(self._parse_gga(line)) # TODO: _parse_gga 함수 완성
                    elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                        rmc_data.append(self._parse_rmc(line)) # TODO: _parse_rmc 함수 완성
        except FileNotFoundError:
            print(f"NMEA 파일을 찾을 수 없습니다: {file_path}")
            print("시뮬레이션 데이터를 생성합니다...")
            return self._generate_nmea_simulation()
        
        return {
            'gga': [data for data in gga_data if data is not None],
            'rmc': [data for data in rmc_data if data is not None]
        }
    
    def _parse_gga(self, line: str) -> Optional[Dict]:
        """
        GGA 메시지를 파싱합니다.
        GGA: $GPGGA,utc,lat,lat_dir,lon,lon_dir,fix_quality,num_satellites,hdop,altitude,altitude_unit,undulation,u_units,age,stn_ID,checksum,[CR][LF]
        """
        try:
            parts = line.split(',')
            if len(parts) < 15:
                return None
                
            # 위도 변환
            lat_str = parts[2] # TODO: 위도 문자열 추출 - parts 사용
            lat_hemisphere = parts[3] # TODO: 위도 방향 추출
            if lat_str and lat_hemisphere:
                lat_deg = int(lat_str[:2])
                lat_min = float(lat_str[2:])
                latitude = lat_deg + lat_min / 60.0
                if lat_hemisphere == 'S':
                    latitude = -latitude
            else:
                latitude = None
                
            # 경도 변환
            lon_str = parts[4] # TODO: 경도 문자열 추출
            lon_hemisphere = parts[5] # TODO: 경도 방향 추출
            if lon_str and lon_hemisphere:
                lon_deg = int(lon_str[:3])
                lon_min = float(lon_str[3:])
                longitude = lon_deg + lon_min / 60.0
                if lon_hemisphere == 'W':
                    longitude = -longitude
            else:
                longitude = None
                
            return {
                'time': parts[1], # TODO: UTC 시간 추출
                'latitude': latitude,
                'longitude': longitude,
                'fix_quality': int(parts[6]) if parts[6] else 0, # TODO: FIX 품질 추출
                'num_satellites': int(parts[7]) if parts[7] else 0, # TODO: 위성 수 추출
                'hdop': float(parts[8]) if parts[8] else 0.0, # TODO: HDOP 추출
                'altitude': float(parts[9]) if parts[9] else 0.0, # TODO: 고도 추출
                'altitude_unit': parts[10] if parts[10] else '', # 고도 단위 추출
                'geoid_height': float(parts[11]) if parts[11] else 0.0 # 지구 고도 추출 (undulation)
            }
        except (ValueError, IndexError):
            return None
    
    def _parse_rmc(self, line: str) -> Optional[Dict]:
        """
        RMC 메시지를 파싱합니다.
        RMC: $GPRMC,utc, pos_status, lat, lat_dir,lon,lon_dir,speed_Knots,track_true,date,mag_var,var_dir,mode_ind,checksum,[CR][LF]
        """
        try:
            parts = line.split(',')
            if len(parts) < 12:
                return None
                
            # 위도 변환
            lat_str = parts[3] # 위도 문자열 추출
            lat_hemisphere = parts[4] # TODO: 위도 방향 추출
            if lat_str and lat_hemisphere:
                lat_deg = int(lat_str[:2])
                lat_min = float(lat_str[2:])
                latitude = lat_deg + lat_min / 60.0
                if lat_hemisphere == 'S':
                    latitude = -latitude
            else:
                latitude = None
                
            # 경도 변환
            lon_str = parts[5] # TODO: 경도 문자열 추출
            lon_hemisphere = parts[6] # TODO: 경도 방향 추출
            if lon_str and lon_hemisphere:
                lon_deg = int(lon_str[:3])
                lon_min = float(lon_str[3:])
                longitude = lon_deg + lon_min / 60.0
                if lon_hemisphere == 'W':
                    longitude = -longitude
            else:
                longitude = None
                
            return {
                'time': parts[1], # TODO: UTC 시간 추출
                'status': parts[2], # TODO: 상태 추출
                'latitude': latitude,
                'longitude': longitude,
                'speed_knots': float(parts[7]) if parts[7] else 0.0, # 속도 추출
                'course': float(parts[8]) if parts[8] else 0.0, # 방향 추출 (Track true)
                'date': parts[9] # 날짜 추출
            }
        except (ValueError, IndexError):
            return None
    
    def _generate_nmea_simulation(self) -> Dict:
        """시뮬레이션 NMEA 데이터를 생성합니다."""
        num_points = 100
        
        # 한양대학교 근처 좌표
        center_lat = 37.5561
        center_lon = 127.0445
        
        # 시뮬레이션 궤적 생성 (원형)
        angles = np.linspace(0, 2*np.pi, num_points)
        radius_lat = 0.0005  # 약 50m
        radius_lon = 0.0005
        
        gga_data = []
        rmc_data = []
        
        for i, angle in enumerate(angles):
            lat = center_lat + radius_lat * np.cos(angle)
            lon = center_lon + radius_lon * np.sin(angle)
            
            # GGA 데이터
            gga_data.append({
                'time': f"{9 + i//60:02d}{(i%60):02d}{(i%60)*10:02d}.00",
                'latitude': lat,
                'longitude': lon,
                'fix_quality': 1,
                'num_satellites': 8 + int(2 * np.sin(i/10)),
                'hdop': 1.0 + 0.5 * np.sin(i/20),
                'altitude': 50.0 + 10 * np.sin(i/15),
                'altitude_unit': 'M',
                'geoid_height': 28.0
            })
            
            # RMC 데이터
            speed = 5.0 + 2.0 * np.sin(i/10)  # 속도 변화
            course = np.degrees(angle) % 360  # 방향
            
            rmc_data.append({
                'time': f"{9 + i//60:02d}{(i%60):02d}{(i%60)*10:02d}.00",
                'status': 'A',
                'latitude': lat,
                'longitude': lon,
                'speed_knots': speed,
                'course': course,
                'date': '050825'
            })
        
        return {'gga': gga_data, 'rmc': rmc_data}
    
    def load_rosbag_gnss_data(self, bag_path: str, topics: List[str]) -> Dict:
        """
        ROS bag 파일에서 GNSS 데이터를 로드합니다.
        
        Args:
            bag_path (str): bag 파일 경로
            topics (List[str]): 로드할 토픽 리스트
            
        Returns:
            Dict: 토픽별 데이터가 담긴 딕셔너리
        """
        data = {topic: {'time': [], 'data': []} for topic in topics}
        
        try:
            if use_rosbag:
                # Ubuntu/Linux 환경에서 rosbag 사용
                with rosbag.Bag(bag_path, 'r') as bag:
                    for topic, msg, t in bag.read_messages(topics=topics):
                        timestamp = t.to_sec()
                        data[topic]['time'].append(timestamp)
                        data[topic]['data'].append(msg)
            elif use_rosbags:
                # Windows 환경에서 rosbags 라이브러리 사용
                bag_path_obj = Path(bag_path)
                with AnyReader([bag_path_obj]) as reader:
                    # 지정된 토픽들에 대한 연결 필터링
                    topic_connections = [conn for conn in reader.connections if conn.topic in topics]
                    
                    # 메시지 읽기
                    for connection, timestamp, rawdata in reader.messages(connections=topic_connections):
                        # 타임스탬프를 초 단위로 변환 (nanoseconds → seconds)
                        timestamp_sec = timestamp / 1e9
                        
                        # 메시지 역직렬화
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        
                        # 데이터 저장
                        data[connection.topic]['time'].append(timestamp_sec)
                        data[connection.topic]['data'].append(msg)
            else:
                raise ImportError("rosbag 또는 rosbags 라이브러리가 설치되지 않았습니다.")
                    
        except Exception as e:
            print(f"ROS bag 로딩 중 오류 발생: {e}")
            print("오류 상세 정보:", str(e))
            print("시뮬레이션 데이터를 생성합니다...")
            return self._generate_rosbag_simulation(topics)
            
        return data
    
    def _generate_rosbag_simulation(self, topics: List[str]) -> Dict:
        """시뮬레이션 ROS bag 데이터를 생성합니다."""
        data = {topic: {'time': [], 'data': []} for topic in topics}
        
        # 시간 생성 (30초간, 0.1초 간격)
        time_points = np.arange(0, 30, 0.1)
        
        # 한양대학교 근처 좌표
        center_lat = 37.5561
        center_lon = 127.0445
        
        for i, t in enumerate(time_points):
            # 원형 궤적
            angle = 0.1 * i
            radius = 0.0005
            lat = center_lat + radius * np.cos(angle)
            lon = center_lon + radius * np.sin(angle)
            
            for topic in topics:
                if 'bestpos' in topic:
                    msg = type('BestPos', (), {})()
                    msg.latitude = lat
                    msg.longitude = lon
                    msg.height = 50.0 + 5 * np.sin(i/10)
                    msg.latitude_std_dev = 0.1 + 0.05 * np.random.random()
                    msg.longitude_std_dev = 0.1 + 0.05 * np.random.random()
                    msg.height_std_dev = 0.2 + 0.1 * np.random.random()
                    msg.sol_status = 'SOL_COMPUTED'
                    msg.pos_type = 'NARROW_INT'
                    
                elif 'bestvel' in topic:
                    msg = type('BestVel', (), {})()
                    msg.horizontal_speed = 2.0 + 1.0 * np.sin(i/20)
                    msg.track_ground = np.degrees(angle) % 360
                    msg.vertical_speed = 0.1 * np.sin(i/15)
                    msg.horizontal_speed_std_dev = 0.05
                    msg.track_ground_std_dev = 1.0
                    msg.vertical_speed_std_dev = 0.1
                    
                elif 'inspvax' in topic:
                    msg = type('InsPvaX', (), {})()
                    msg.latitude = lat
                    msg.longitude = lon
                    msg.height = 50.0 + 5 * np.sin(i/10)
                    msg.north_velocity = 2.0 * np.cos(angle)
                    msg.east_velocity = 2.0 * np.sin(angle)
                    msg.up_velocity = 0.1 * np.sin(i/15)
                    msg.azimuth = np.degrees(angle) % 360
                    msg.latitude_stdev = 0.1
                    msg.longitude_stdev = 0.1
                    msg.height_stdev = 0.2
                    
                elif '/gps/gps' in topic:
                    msg = type('GPSFix', (), {})()
                    # 기본 위치 정보
                    msg.latitude = lat
                    msg.longitude = lon
                    msg.altitude = 48.0 + 5 * np.sin(i/10)
                    msg.track = np.degrees(angle) % 360  # heading
                    msg.speed = 2.5 + 1.5 * np.sin(i/20)  # m/s
                    msg.climb = 0.05 * np.sin(i/15)  # vertical speed
                    
                    # DOP 값들 (시뮬레이션)
                    msg.gdop = 2.5 + 0.5 * np.sin(i/30)
                    msg.pdop = 1.8 + 0.3 * np.sin(i/25)
                    msg.hdop = 0.8 + 0.2 * np.sin(i/20)
                    msg.vdop = 1.6 + 0.3 * np.sin(i/18)
                    msg.tdop = 1.2 + 0.2 * np.sin(i/22)
                    
                    # 오차 정보
                    msg.err = 0.2 + 0.05 * np.random.random()
                    msg.err_horz = 0.25 + 0.1 * np.random.random()
                    msg.err_vert = 0.18 + 0.08 * np.random.random()
                    
                    # Position covariance (3x3 matrix를 1차원 배열로)
                    # [xx, xy, xz, yx, yy, yz, zx, zy, zz] 순서
                    cov_xx = (msg.err_horz * 0.001) ** 2  # longitude variance
                    cov_yy = (msg.err_horz * 0.001) ** 2  # latitude variance  
                    cov_zz = (msg.err_vert * 0.001) ** 2  # altitude variance
                    msg.position_covariance = [cov_xx, 0.0, 0.0, 0.0, cov_yy, 0.0, 0.0, 0.0, cov_zz]
                    msg.position_covariance_type = 2  # COVARIANCE_TYPE_DIAGONAL_KNOWN
                    
                    # Status 정보
                    status = type('GPSStatus', (), {})()
                    status.satellites_used = 15 + int(3 * np.sin(i/15))
                    status.satellites_visible = status.satellites_used + 2
                    status.status = 18  # STATUS_GBAS_FIX
                    msg.status = status
                    
                data[topic]['time'].append(t)
                data[topic]['data'].append(msg)
        
        return data
    
    def meridional_radius(self, ref_latitude_deg: float) -> float:
        """자오선 곡률반지름을 계산합니다."""
        lat = np.deg2rad(ref_latitude_deg)
        
        a = self.WGS84_A
        b = self.WGS84_B
        
        num = (a * b) ** 2 # 자오선 곡률반지름 계산식
        den = ((a * np.cos(lat)) ** 2 + (b * np.sin(lat)) ** 2) ** (3/2) # 자오선 곡률반지름 계산식
        
        return num / den # 자오선 곡률반지름 계산식 완성
    
    def normal_radius(self, ref_latitude_deg: float) -> float:
        """수직 곡률반지름을 계산합니다."""
        lat = np.deg2rad(ref_latitude_deg)
        
        a = self.WGS84_A
        b = self.WGS84_B
        
        return a * a / np.sqrt((a * np.cos(lat)) ** 2 + (b * np.sin(lat)) ** 2) # 수직 곡률반지름 계산식 완성
    
    def wgs84_to_enu(self, llh: np.ndarray, ref_llh: np.ndarray) -> np.ndarray:
        """
        WGS84 좌표를 ENU 좌표로 변환합니다.
        
        Args:
            llh (np.ndarray): 변환할 좌표 [lat, lon, height] (N x 3)
            ref_llh (np.ndarray): 기준점 좌표 [lat, lon, height]
            
        Returns:
            np.ndarray: ENU 좌표 [east, north, up] (N x 3)
        """
        enu = np.zeros_like(llh)
        
        ref_height = ref_llh[2]
        ref_lat = np.deg2rad(ref_llh[0])
        
        meridional_r = self.meridional_radius(ref_llh[0]) # TODO: meridional_radius 함수 사용
        normal_r = self.normal_radius(ref_llh[0]) # TODO: normal_radius 함수 사용
        
        for idx in range(llh.shape[0]):
            delta_lat_deg = llh[idx, 0] - ref_llh[0] # TODO: 위도 차이 계산
            delta_lon_deg = llh[idx, 1] - ref_llh[1] # TODO: 경도 차이 계산
            
            enu[idx, 0] = np.deg2rad(delta_lon_deg) * (normal_r + ref_height) * np.cos(ref_lat) # East 추출
            enu[idx, 1] = np.deg2rad(delta_lat_deg) * (meridional_r + ref_height)  # North 추출
            enu[idx, 2] = llh[idx, 2] - ref_llh[2] # TODO: Up 추출
            
        return enu
    
    def enu_to_wgs84(self, enu: np.ndarray, ref_llh: np.ndarray) -> np.ndarray:
        """
        ENU 좌표를 WGS84 좌표로 변환합니다.
        
        Args:
            enu (np.ndarray): 변환할 ENU 좌표 [east, north, up] (N x 3)
            ref_llh (np.ndarray): 기준점 좌표 [lat, lon, height]
            
        Returns:
            np.ndarray: WGS84 좌표 [lat, lon, height] (N x 3)
        """
        llh = np.zeros_like(enu)
        
        ref_height = ref_llh[2]
        ref_lat = np.deg2rad(ref_llh[0])
        
        meridional_r = self.meridional_radius(ref_llh[0]) # TODO: meridional_radius 함수 사용
        normal_r = self.normal_radius(ref_llh[0]) # TODO: normal_radius 함수 사용
        
        for idx in range(enu.shape[0]):
            delta_lat_deg = np.rad2deg(enu[idx, 1] / (meridional_r + ref_height)) # 위도 차이 계산
            delta_lon_deg = np.rad2deg(enu[idx, 0] / ((normal_r + ref_height) * np.cos(ref_lat))) # 경도 차이 계산
            
            # 기준점으로부터 업데이트
            llh[idx, 0] = ref_llh[0] + delta_lat_deg # TODO: Latitude 추출
            llh[idx, 1] = ref_llh[1] + delta_lon_deg # TODO: Longitude 추출
            llh[idx, 2] = ref_llh[2] + enu[idx, 2] # TODO: Height 추출
            
        return llh
    
    def plot_nmea_analysis(self, nmea_data: Dict):
        """NMEA 데이터 분석 시각화"""
        gga_data = nmea_data['gga']
        rmc_data = nmea_data['rmc']
        
        if not gga_data or not rmc_data:
            print("유효한 NMEA 데이터가 없습니다.")
            return
        
        # 데이터 추출
        latitudes = [d['latitude'] for d in gga_data if d['latitude'] is not None]
        longitudes = [d['longitude'] for d in gga_data if d['longitude'] is not None]
        hdops = [d['hdop'] for d in gga_data if d['hdop'] is not None]
        num_sats = [d['num_satellites'] for d in gga_data if d['num_satellites'] is not None]
        speeds = [d['speed_knots'] for d in rmc_data if d['speed_knots'] is not None]
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 궤적
        axes[0, 0].plot(longitudes, latitudes, 'b-', linewidth=2)
        axes[0, 0].plot(longitudes[0], latitudes[0], 'go', markersize=10, label='Start')
        axes[0, 0].plot(longitudes[-1], latitudes[-1], 'ro', markersize=10, label='End')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_title('GPS Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # HDOP
        axes[0, 1].plot(hdops, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time Index')
        axes[0, 1].set_ylabel('HDOP')
        axes[0, 1].set_title('Horizontal Dilution of Precision')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 위성 수
        axes[0, 2].plot(num_sats, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Time Index')
        axes[0, 2].set_ylabel('Number of Satellites')
        axes[0, 2].set_title('Satellite Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 속도
        axes[1, 0].plot(speeds, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Speed (knots)')
        axes[1, 0].set_title('Ground Speed')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HDOP 히스토그램
        axes[1, 1].hist(hdops, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('HDOP')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('HDOP Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 위성 수 히스토그램
        axes[1, 2].hist(num_sats, bins=range(min(num_sats)-5, max(num_sats)+5), 
                       alpha=0.7, color='cyan', edgecolor='black')
        axes[1, 2].set_xlabel('Number of Satellites')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Satellite Count Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 통계 출력
        print("=== NMEA 데이터 분석 결과 ===")
        print(f"총 GGA 메시지 수: {len(gga_data)}")
        print(f"총 RMC 메시지 수: {len(rmc_data)}")
        print(f"위도 범위: {min(latitudes):.6f} ~ {max(latitudes):.6f}")
        print(f"경도 범위: {min(longitudes):.6f} ~ {max(longitudes):.6f}")
        print(f"HDOP 범위: {min(hdops):.2f} ~ {max(hdops):.2f}")
        print(f"위성 수 범위: {min(num_sats)} ~ {max(num_sats)}개")
        print(f"속도 범위: {min(speeds):.2f} ~ {max(speeds):.2f} knots")
    
    def create_satellite_map(self, latitudes: List[float], longitudes: List[float], 
                           times: Optional[List[float]] = None, 
                           headings: Optional[List[float]] = None,
                           title: str = "GPS Trajectory") -> folium.Map:
        """
        위성 지도 위에 GPS 궤적을 시각화합니다.
        
        Args:
            latitudes: 위도 리스트
            longitudes: 경도 리스트
            times: 시간 리스트 (선택사항)
            headings: 헤딩 리스트 (선택사항)
            title: 지도 제목
            
        Returns:
            folium.Map: 생성된 지도 객체
        """
        if not latitudes or not longitudes:
            print("유효한 좌표 데이터가 없습니다.")
            return None
        
        # 지도 중심점 계산
        center_lat = np.mean(latitudes)
        center_lon = np.mean(longitudes)
        
        # 지도 생성
        m = folium.Map(
            location=[np.mean(latitudes), np.mean(longitudes)], # TODO: 지도 중심점 대입
            zoom_start=18,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri WorldImagery'
        )
        
        # 궤적 그리기
        if times is not None:
            # 시간에 따른 색상 변화
            normalized_times = (np.array(times) - times[0]) / (times[-1] - times[0])
            colors = plt.cm.jet(normalized_times)
            
            for i in range(len(latitudes)-1):
                color = f"#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}"
                folium.PolyLine(
                    locations=[[latitudes[i], longitudes[i]], [latitudes[i+1], longitudes[i+1]]],
                    color=color,
                    weight=3,
                    opacity=0.8
                ).add_to(m)
        else:
            # 단색 궤적
            folium.PolyLine(
                locations=list(zip(latitudes, longitudes)),
                color='red',
                weight=3,
                opacity=0.8
            ).add_to(m)
        
        # 시작점과 끝점 마커
        folium.Marker(
            location=[latitudes[0], longitudes[0]], # TODO: 시작점 좌표 대입 - latitudes[xxxxxx], longitudes[xxxxxx]
            popup="Start Point",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        folium.Marker(
            location=[latitudes[-1], longitudes[-1]], # TODO: 끝점 좌표 대입 - latitudes[xxxxxx], longitudes[xxxxxx]
            popup="End Point",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
        
        # 헤딩 화살표 추가
        if headings is not None:
            step = max(1, len(latitudes) // 20)  # 최대 20개의 화살표
            for i in range(0, len(latitudes), step):
                if i < len(headings):
                    # 간단한 화살표 마커
                    folium.RegularPolygonMarker(
                        location=[latitudes[i], longitudes[i]],
                        number_of_sides=3,
                        radius=5,
                        rotation=headings[i],
                        color='black',
                        fill_color='black',
                        fillOpacity=0.7,
                        popup=f"Heading: {headings[i]:.1f}°"
                    ).add_to(m)
        
        return m
    
    def plot_gps_fix_analysis(self, rosbag_data: Dict):
        """GPS Fix 데이터 분석 시각화"""
        if '/gps/gps' not in rosbag_data:
            print("GPS Fix 데이터를 찾을 수 없습니다.")
            return
            
        gps_data = rosbag_data['/gps/gps']
        times = np.array(gps_data['time'])
        relative_times = times - times[0]
        
        # 데이터 추출
        latitudes = [msg.latitude for msg in gps_data['data']]
        longitudes = [msg.longitude for msg in gps_data['data']]
        altitudes = [msg.altitude for msg in gps_data['data']]
        speeds = [msg.speed for msg in gps_data['data']]
        tracks = [msg.track for msg in gps_data['data']]
        # track 값과 altitude 값에는 보정으로 인한 데이터 손실이 중간에 존재하므로, 앞뒤 데이터 차이보다 100배 이상 차이날 경우, 앞뒤 데이터의 평균으로 대체
        for i in range(2, len(tracks)-2):
            if abs(tracks[i] - tracks[i-2]) > 100 * abs(tracks[i+2] - tracks[i-2]):
                tracks[i] = (tracks[i-2] + tracks[i+2]) / 2
        
        for i in range(2, len(altitudes)-2):
            if abs(altitudes[i] - altitudes[i-2]) > 100 * abs(altitudes[i+2] - altitudes[i-2]):
                altitudes[i] = (altitudes[i-2] + altitudes[i+2]) / 2
        
        # DOP 값들
        gdops = [msg.gdop for msg in gps_data['data']]
        pdops = [msg.pdop for msg in gps_data['data']]
        hdops = [msg.hdop for msg in gps_data['data']]
        vdops = [msg.vdop for msg in gps_data['data']]
        tdops = [msg.tdop for msg in gps_data['data']]
        
        # 오차 정보
        err_horzs = [msg.err_horz for msg in gps_data['data']]
        err_verts = [msg.err_vert for msg in gps_data['data']]
        
        # 위성 수
        satellites_used = [msg.status.satellites_used for msg in gps_data['data']]
        satellites_visible = [msg.status.satellites_visible for msg in gps_data['data']]
        
        # 공분산 대각 성분 추출
        pos_covs_xx = [msg.position_covariance[0] for msg in gps_data['data']]
        pos_covs_yy = [msg.position_covariance[4] for msg in gps_data['data']]
        pos_covs_zz = [msg.position_covariance[8] for msg in gps_data['data']]
        
        # 시각화
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 궤적
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(longitudes, latitudes, 'b-', linewidth=2)
        plt.plot(longitudes[0], latitudes[0], 'go', markersize=10, label='Start')
        plt.plot(longitudes[-1], latitudes[-1], 'ro', markersize=10, label='End')
        plt.xlabel('Longitude (deg)')
        plt.ylabel('Latitude (deg)')
        plt.title('GPS Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. DOP 값들
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(relative_times, gdops, 'r-', label='GDOP', linewidth=2)
        plt.plot(relative_times, pdops, 'g-', label='PDOP', linewidth=2)
        plt.plot(relative_times, hdops, 'b-', label='HDOP', linewidth=2)
        plt.plot(relative_times, vdops, 'm-', label='VDOP', linewidth=2)
        plt.plot(relative_times, tdops, 'c-', label='TDOP', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('DOP')
        plt.title('Dilution of Precision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 수평/수직 오차
        ax3 = plt.subplot(3, 3, 3)
        plt.plot(relative_times, err_horzs, 'r-', label='Horizontal Error', linewidth=2)
        plt.plot(relative_times, err_verts, 'b-', label='Vertical Error', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m)')
        plt.title('Position Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 속도
        ax4 = plt.subplot(3, 3, 4)
        plt.plot(relative_times, speeds, 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('Ground Speed')
        plt.grid(True, alpha=0.3)
        
        # 5. 방향
        ax5 = plt.subplot(3, 3, 5)
        plt.plot(relative_times, tracks, 'orange', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Track (deg)')
        plt.title('Ground Track (Heading)')
        plt.grid(True, alpha=0.3)
        
        # 6. 고도
        ax6 = plt.subplot(3, 3, 6)
        plt.plot(relative_times, altitudes, 'm-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.title('Altitude Profile')
        plt.grid(True, alpha=0.3)
        
        # 7. 위성 수
        ax7 = plt.subplot(3, 3, 7)
        plt.plot(relative_times, satellites_used, 'b-', label='Used', linewidth=2)
        plt.plot(relative_times, satellites_visible, 'r--', label='Visible', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Satellites')
        plt.title('Satellite Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. 위치 공분산
        ax8 = plt.subplot(3, 3, 8)
        plt.plot(relative_times, np.sqrt(pos_covs_xx), 'r-', label='Lon σ (m)', linewidth=2)
        plt.plot(relative_times, np.sqrt(pos_covs_yy), 'g-', label='Lat σ (m)', linewidth=2)
        plt.plot(relative_times, np.sqrt(pos_covs_zz), 'b-', label='Alt σ (m)', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Position StdDev (m)')
        plt.title('Position Covariance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 통계 출력
        print("=== GPS Fix 데이터 분석 결과 ===")
        print(f"총 데이터 포인트: {len(gps_data['data'])}개")
        print(f"시간 범위: {relative_times[-1]:.1f}초")
        print(f"위도 범위: {min(latitudes):.8f}° ~ {max(latitudes):.8f}°")
        print(f"경도 범위: {min(longitudes):.8f}° ~ {max(longitudes):.8f}°")
        print(f"고도 범위: {min(altitudes):.1f}m ~ {max(altitudes):.1f}m")
        print(f"속도 범위: {min(speeds):.1f} ~ {max(speeds):.1f} m/s")
        print(f"HDOP 범위: {min(hdops):.2f} ~ {max(hdops):.2f}")
        print(f"수평 오차 범위: {min(err_horzs):.3f} ~ {max(err_horzs):.3f} m")
        print(f"수직 오차 범위: {min(err_verts):.3f} ~ {max(err_verts):.3f} m")
        print(f"사용 위성 수 범위: {min(satellites_used)} ~ {max(satellites_used)}개")
    
    def plot_wgs84_to_enu_comparison(self, original_llh: np.ndarray, converted_enu: np.ndarray, gt_enu: np.ndarray):
        """좌표 변환 결과 비교 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 원본 WGS84
        axes[0].plot(original_llh[:, 1], original_llh[:, 0], 'bo-', 
                       linewidth=2, markersize=4, label='Original WGS84')
        axes[0].set_xlabel('Longitude (deg)')
        axes[0].set_ylabel('Latitude (deg)')
        axes[0].set_title('Original WGS84 Coordinates')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # ENU 좌표
        axes[1].plot(gt_enu[:, 0], gt_enu[:, 1], 'go-', 
                       linewidth=2, markersize=8, label='GT ENU')
        axes[1].plot(converted_enu[:, 0], converted_enu[:, 1], 'ro-', 
                       linewidth=2, markersize=4, label='Converted ENU')
        axes[1].set_xlabel('East (m)')
        axes[1].set_ylabel('North (m)')
        axes[1].set_title('ENU Coordinates')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].axis('equal')
        
        # 오차 분석
        lat_error = gt_enu[:, 0] - converted_enu[:, 0]
        lon_error = gt_enu[:, 1] - converted_enu[:, 1]
        height_error = gt_enu[:, 2] - converted_enu[:, 2]
        
        axes[2].plot(lat_error, 'r-', label='Latitude Error (m)', linewidth=2)
        axes[2].plot(lon_error, 'g-', label='Longitude Error (m)', linewidth=2)
        axes[2].plot(height_error, 'b-', label='Height Error (m)', linewidth=2)
        axes[2].set_xlabel('Point Index')
        axes[2].set_ylabel('Error')
        axes[2].set_title('Conversion Error Analysis')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 통계 출력
        print("=== 좌표 변환 정확도 분석 ===")
        print(f"위도 오차 RMS: {np.sqrt(np.mean(lat_error**2)):.3f} m")
        print(f"경도 오차 RMS: {np.sqrt(np.mean(lon_error**2)):.3f} m")
        print(f"고도 오차 RMS: {np.sqrt(np.mean(height_error**2)):.3f} m")
        print(f"수평 오차 RMS: {np.sqrt(np.mean(lat_error**2 + lon_error**2)):.3f} m")

    def plot_enu_to_wgs84_comparison(self, original_enu: np.ndarray, converted_wgs84: np.ndarray, gt_llh: np.ndarray):
        """WGS84 to ENU 좌표 변환 결과 비교 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 원본 ENU
        axes[0].plot(original_enu[:, 0], original_enu[:, 1], 'bo-', 
                       linewidth=2, markersize=4, label='Original ENU')
        axes[0].set_xlabel('East (m)')
        axes[0].set_ylabel('North (m)')
        axes[0].set_title('Original ENU Coordinates')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # WGS84 좌표
        axes[1].plot(gt_llh[:, 1], gt_llh[:, 0], 'go-', 
                       linewidth=2, markersize=8, label='GT WGS84')
        axes[1].plot(converted_wgs84[:, 1], converted_wgs84[:, 0], 'ro-', 
                       linewidth=2, markersize=4, label='Converted WGS84')
        axes[1].set_xlabel('Longitude (deg)')
        axes[1].set_ylabel('Latitude (deg)')
        axes[1].set_title('WGS84 Coordinates')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].axis('equal')

        # 오차 분석
        lat_error = gt_llh[:, 0] - converted_wgs84[:, 0]
        lon_error = gt_llh[:, 1] - converted_wgs84[:, 1]
        height_error = gt_llh[:, 2] - converted_wgs84[:, 2]
        
        axes[2].plot(lat_error, 'r-', label='Latitude Error (m)', linewidth=2)
        axes[2].plot(lon_error, 'g-', label='Longitude Error (m)', linewidth=2)
        axes[2].plot(height_error, 'b-', label='Height Error (m)', linewidth=2)
        axes[2].set_xlabel('Point Index')
        axes[2].set_ylabel('Error')
        axes[2].set_title('Conversion Error Analysis')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)