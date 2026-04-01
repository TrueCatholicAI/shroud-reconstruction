"""
Compute facial proportion ratios from Study 1 and Study 2 landmarks.
Ratios are scale-independent for cross-study comparison.
"""
import json
import math

def euclidean(p1, p2):
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def compute_ratios_s1(landmarks):
    """Compute ratios for Study 1 (Enrie 1931)"""
    k = landmarks['key_landmarks']
    
    # Core measurements
    ipd = euclidean(k['left_pupil'], k['right_pupil'])
    face_width = abs(k['right_cheek']['x'] - k['left_cheek']['x'])
    nose_width = abs(k['nose_right_alar']['x'] - k['nose_left_alar']['x'])
    mouth_width = abs(k['mouth_right']['x'] - k['mouth_left']['x'])
    nose_to_chin = abs(k['chin']['y'] - k['nose_tip']['y'])
    jaw_width = abs(k['jaw_right']['x'] - k['jaw_left']['x'])
    # Face height: brow/bridge to chin
    face_height = abs(k['chin']['y'] - k['nose_bridge_top']['y'])
    
    return {
        'ipd': ipd,
        'face_width': face_width,
        'nose_width': nose_width,
        'mouth_width': mouth_width,
        'nose_to_chin': nose_to_chin,
        'jaw_width': jaw_width,
        'face_height': face_height
    }

def compute_ratios_s2(landmarks):
    """Compute ratios for Study 2 (Miller 1978)"""
    # Study 2 landmarks are in 150x150 coordinates
    ipd = euclidean(landmarks['left_pupil'], landmarks['right_pupil'])
    face_width = abs(landmarks['right_cheek']['x'] - landmarks['left_cheek']['x'])
    nose_width = abs(landmarks['nose_right_alar']['x'] - landmarks['nose_left_alar']['x'])
    mouth_width = abs(landmarks['mouth_right']['x'] - landmarks['mouth_left']['x'])
    nose_to_chin = abs(landmarks['chin']['y'] - landmarks['nose_tip']['y'])
    jaw_width = abs(landmarks['right_jaw']['x'] - landmarks['left_jaw']['x'])
    # Face height: brow_center to chin
    face_height = abs(landmarks['chin']['y'] - landmarks['brow_center']['y'])
    
    return {
        'ipd': ipd,
        'face_width': face_width,
        'nose_width': nose_width,
        'mouth_width': mouth_width,
        'nose_to_chin': nose_to_chin,
        'jaw_width': jaw_width,
        'face_height': face_height
    }

def main():
    # Load Study 1 landmarks
    with open('data/measurements/landmarks.json', 'r') as f:
        s1_landmarks = json.load(f)
    
    # Load Study 2 landmarks
    with open('output/study2_miller/landmarks.json', 'r') as f:
        s2_landmarks = json.load(f)
    
    # Compute raw measurements
    s1_raw = compute_ratios_s1(s1_landmarks)
    s2_raw = compute_ratios_s2(s2_landmarks)
    
    # Compute ratios
    # IPD / Face Width
    s1_ipd_fw = s1_raw['ipd'] / s1_raw['face_width'] if s1_raw['face_width'] > 0 else 0
    s2_ipd_fw = s2_raw['ipd'] / s2_raw['face_width'] if s2_raw['face_width'] > 0 else 0
    
    # Nose Width / Face Width
    s1_nw_fw = s1_raw['nose_width'] / s1_raw['face_width'] if s1_raw['face_width'] > 0 else 0
    s2_nw_fw = s2_raw['nose_width'] / s2_raw['face_width'] if s2_raw['face_width'] > 0 else 0
    
    # Mouth Width / Face Width
    s1_mw_fw = s1_raw['mouth_width'] / s1_raw['face_width'] if s1_raw['face_width'] > 0 else 0
    s2_mw_fw = s2_raw['mouth_width'] / s2_raw['face_width'] if s2_raw['face_width'] > 0 else 0
    
    # Nose-to-Chin / Face Height
    s1_nc_fh = s1_raw['nose_to_chin'] / s1_raw['face_height'] if s1_raw['face_height'] > 0 else 0
    s2_nc_fh = s2_raw['nose_to_chin'] / s2_raw['face_height'] if s2_raw['face_height'] > 0 else 0
    
    # Jaw Width / Face Width
    s1_jw_fw = s1_raw['jaw_width'] / s1_raw['face_width'] if s1_raw['face_width'] > 0 else 0
    s2_jw_fw = s2_raw['jaw_width'] / s2_raw['face_width'] if s2_raw['face_width'] > 0 else 0
    
    # Face Height / Face Width (aspect ratio)
    s1_fh_fw = s1_raw['face_height'] / s1_raw['face_width'] if s1_raw['face_width'] > 0 else 0
    s2_fh_fw = s2_raw['face_height'] / s2_raw['face_width'] if s2_raw['face_width'] > 0 else 0
    
    results = {
        'study1_raw': s1_raw,
        'study2_raw': s2_raw,
        'ratios': {
            'ipd_face_width': {
                'study1': round(s1_ipd_fw, 4),
                'study2': round(s2_ipd_fw, 4),
                'difference_pct': round(abs(s1_ipd_fw - s2_ipd_fw) / max(s1_ipd_fw, s2_ipd_fw) * 100, 1) if max(s1_ipd_fw, s2_ipd_fw) > 0 else 0
            },
            'nose_width_face_width': {
                'study1': round(s1_nw_fw, 4),
                'study2': round(s2_nw_fw, 4),
                'difference_pct': round(abs(s1_nw_fw - s2_nw_fw) / max(s1_nw_fw, s2_nw_fw) * 100, 1) if max(s1_nw_fw, s2_nw_fw) > 0 else 0
            },
            'mouth_width_face_width': {
                'study1': round(s1_mw_fw, 4),
                'study2': round(s2_mw_fw, 4),
                'difference_pct': round(abs(s1_mw_fw - s2_mw_fw) / max(s1_mw_fw, s2_mw_fw) * 100, 1) if max(s1_mw_fw, s2_mw_fw) > 0 else 0
            },
            'nose_to_chin_face_height': {
                'study1': round(s1_nc_fh, 4),
                'study2': round(s2_nc_fh, 4),
                'difference_pct': round(abs(s1_nc_fh - s2_nc_fh) / max(s1_nc_fh, s2_nc_fh) * 100, 1) if max(s1_nc_fh, s2_nc_fh) > 0 else 0
            },
            'jaw_width_face_width': {
                'study1': round(s1_jw_fw, 4),
                'study2': round(s2_jw_fw, 4),
                'difference_pct': round(abs(s1_jw_fw - s2_jw_fw) / max(s1_jw_fw, s2_jw_fw) * 100, 1) if max(s1_jw_fw, s2_jw_fw) > 0 else 0
            },
            'face_height_face_width': {
                'study1': round(s1_fh_fw, 4),
                'study2': round(s2_fh_fw, 4),
                'difference_pct': round(abs(s1_fh_fw - s2_fh_fw) / max(s1_fh_fw, s2_fh_fw) * 100, 1) if max(s1_fh_fw, s2_fh_fw) > 0 else 0
            }
        },
        'summary': {
            'mean_difference_pct': round(sum(r['difference_pct'] for r in results['ratios'].values()) / len(results['ratios']), 1) if 'results' in dir() else 0,
            'note': 'Lower difference percentages indicate more consistent proportions between studies'
        }
    }
    
    # Recalculate mean properly
    diffs = [r['difference_pct'] for r in results['ratios'].values()]
    results['summary']['mean_difference_pct'] = round(sum(diffs) / len(diffs), 1)
    
    # Save results
    output_path = 'output/task_results/ratio_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print("\n=== RATIO COMPARISON ===")
    for name, data in results['ratios'].items():
        print(f"{name}: S1={data['study1']}, S2={data['study2']}, diff={data['difference_pct']}%")
    print(f"\nMean difference: {results['summary']['mean_difference_pct']}%")

if __name__ == '__main__':
    main()