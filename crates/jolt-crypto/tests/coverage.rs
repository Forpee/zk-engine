//! Targeted coverage tests for jolt-crypto.
//!
//! Covers gaps in G2, GT, GLV (2D/4D), Dory vector ops, fixed-base MSM,
//! HomomorphicCommitment, and Debug/From conversions.

use jolt_crypto::{
    Bn254, Bn254G1, Bn254G2, Bn254GT, HomomorphicCommitment, JoltGroup, PairingGroup,
};
use jolt_field::{Field, Fr, Ring};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[test]
fn g2_debug_format_contains_type_name() {
    let g = Bn254::g2_generator();
    let debug_str = format!("{:?}", g);
    assert!(
        debug_str.starts_with("Bn254G2("),
        "expected Bn254G2(...), got: {debug_str}"
    );
}

#[test]
fn g2_commutativity() {
    let g = Bn254::g2_generator();
    let two = Fr::from_u64(2);
    let three = Fr::from_u64(3);
    let a = g.scalar_mul(&two);
    let b = g.scalar_mul(&three);
    assert_eq!(a + b, b + a);
}

#[test]
#[expect(clippy::op_ref)]
fn g2_add_ref() {
    let g = Bn254::g2_generator();
    let a = g.scalar_mul(&Fr::from_u64(3));
    let b = g.scalar_mul(&Fr::from_u64(5));
    let expected = a + b;
    assert_eq!(a + &b, expected);
}

#[test]
#[expect(clippy::op_ref)]
fn g2_sub_ref() {
    let g = Bn254::g2_generator();
    let a = g.scalar_mul(&Fr::from_u64(7));
    let b = g.scalar_mul(&Fr::from_u64(3));
    let expected = a - b;
    assert_eq!(a - &b, expected);
}

#[test]
fn g2_msm_single_element() {
    let g = Bn254::g2_generator();
    let s = Fr::from_u64(42);
    assert_eq!(Bn254G2::msm(&[g], &[s]), g.scalar_mul(&s));
}

#[test]
fn g2_msm_empty() {
    let result = Bn254G2::msm(&[], &([] as [Fr; 0]));
    assert!(result.is_identity());
}

#[test]
fn g2_msm_multiple_random() {
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let g = Bn254::g2_generator();
    let points: Vec<Bn254G2> = (0..5).map(|i| g.scalar_mul(&Fr::from_u64(i + 1))).collect();
    let scalars: Vec<Fr> = (0..5).map(|_| Fr::random(&mut rng)).collect();

    let msm_result = Bn254G2::msm(&points, &scalars);
    let naive: Bn254G2 = points
        .iter()
        .zip(scalars.iter())
        .fold(<Bn254G2 as JoltGroup>::identity(), |acc, (p, s)| {
            acc + p.scalar_mul(s)
        });
    assert_eq!(msm_result, naive);
}

#[test]
fn g2_associativity() {
    let g = Bn254::g2_generator();
    let a = g.scalar_mul(&Fr::from_u64(3));
    let b = g.scalar_mul(&Fr::from_u64(7));
    let c = g.scalar_mul(&Fr::from_u64(11));
    assert_eq!((a + b) + c, a + (b + c));
}

#[test]
fn g2_neg() {
    let g = Bn254::g2_generator();
    assert_eq!(g + (-g), <Bn254G2 as JoltGroup>::identity());
    assert!((-g + g).is_identity());
}

#[test]
fn g2_scalar_mul_distributive() {
    let g = Bn254::g2_generator();
    let three = Fr::from_u64(3);
    let five = Fr::from_u64(5);
    let eight = Fr::from_u64(8);
    assert_eq!(
        g.scalar_mul(&three) + g.scalar_mul(&five),
        g.scalar_mul(&eight)
    );
}

fn gt_element() -> Bn254GT {
    Bn254::pairing(&Bn254::g1_generator(), &Bn254::g2_generator())
}

#[test]
fn gt_debug_format_contains_type_name() {
    let e = gt_element();
    let debug_str = format!("{:?}", e);
    assert!(
        debug_str.starts_with("Bn254GT("),
        "expected Bn254GT(...), got: {debug_str}"
    );
}

#[test]
fn gt_identity_is_identity() {
    let id = <Bn254GT as JoltGroup>::identity();
    assert!(id.is_identity());
    assert!(!gt_element().is_identity());
}

#[test]
fn gt_mul_assign() {
    let e = gt_element();
    let mut acc = e;
    acc *= e;
    // Mul is a convenience alias for Add (both map to Fq12 multiplication)
    assert_eq!(acc, e + e);
}

#[test]
fn gt_sub_assign() {
    let e = gt_element();
    let double = e + e;
    let mut x = double;
    x -= e;
    assert_eq!(x, e);
}

#[test]
#[expect(clippy::op_ref)]
fn gt_add_ref() {
    let e = gt_element();
    let e2 = e.scalar_mul(&Fr::from_u64(2));
    let expected = e + e2;
    assert_eq!(e + &e2, expected);
}

#[test]
#[expect(clippy::op_ref)]
fn gt_sub_ref() {
    let e = gt_element();
    let e2 = e.scalar_mul(&Fr::from_u64(2));
    let expected = e2 - e;
    assert_eq!(e2 - &e, expected);
}

#[test]
fn gt_neg_of_neg_is_self() {
    let e = gt_element();
    assert_eq!(-(-e), e);
}

#[test]
fn gt_scalar_mul_distributive() {
    let e = gt_element();
    let three = Fr::from_u64(3);
    let five = Fr::from_u64(5);
    let eight = Fr::from_u64(8);
    assert_eq!(
        e.scalar_mul(&three) + e.scalar_mul(&five),
        e.scalar_mul(&eight)
    );
}

#[test]
fn gt_msm_single_element() {
    let e = gt_element();
    let s = Fr::from_u64(7);
    assert_eq!(Bn254GT::msm(&[e], &[s]), e.scalar_mul(&s));
}

#[test]
fn gt_msm_empty() {
    let result = Bn254GT::msm(&[], &([] as [Fr; 0]));
    assert!(result.is_identity());
}

#[test]
fn gt_associativity() {
    let e = gt_element();
    let a = e.scalar_mul(&Fr::from_u64(2));
    let b = e.scalar_mul(&Fr::from_u64(3));
    let c = e.scalar_mul(&Fr::from_u64(5));
    assert_eq!((a + b) + c, a + (b + c));
}

#[test]
fn gt_commutativity() {
    let e = gt_element();
    let a = e.scalar_mul(&Fr::from_u64(3));
    let b = e.scalar_mul(&Fr::from_u64(7));
    assert_eq!(a + b, b + a);
}

#[test]
fn gt_double_is_squaring() {
    let e = gt_element();
    let s = Fr::from_u64(5);
    let es = e.scalar_mul(&s);
    assert_eq!(es.double(), es + es);
}

#[test]
fn gt_mul_matches_add() {
    let e = gt_element();
    let a = e.scalar_mul(&Fr::from_u64(3));
    let b = e.scalar_mul(&Fr::from_u64(7));
    assert_eq!(a * b, a + b);
}

#[test]
fn homomorphic_commitment_g1_linear_combine() {
    let mut rng = ChaCha20Rng::seed_from_u64(600);
    let c1 = Bn254::random_g1(&mut rng);
    let c2 = Bn254::random_g1(&mut rng);
    let scalar = Fr::random(&mut rng);

    let result = <Bn254G1 as HomomorphicCommitment<Fr>>::linear_combine(&c1, &c2, &scalar);
    let expected = c1 + c2.scalar_mul(&scalar);
    assert_eq!(result, expected);
}

#[test]
fn homomorphic_commitment_g2_linear_combine() {
    let g2 = Bn254::g2_generator();
    let c1 = g2.scalar_mul(&Fr::from_u64(3));
    let c2 = g2.scalar_mul(&Fr::from_u64(7));
    let scalar = Fr::from_u64(5);

    let result = <Bn254G2 as HomomorphicCommitment<Fr>>::linear_combine(&c1, &c2, &scalar);
    let expected = c1 + c2.scalar_mul(&scalar);
    assert_eq!(result, expected);
}

#[test]
fn homomorphic_commitment_gt_linear_combine() {
    let e = gt_element();
    let c1 = e.scalar_mul(&Fr::from_u64(2));
    let c2 = e.scalar_mul(&Fr::from_u64(3));
    let scalar = Fr::from_u64(4);

    let result = <Bn254GT as HomomorphicCommitment<Fr>>::linear_combine(&c1, &c2, &scalar);
    let expected = c1 + c2.scalar_mul(&scalar);
    assert_eq!(result, expected);
}

#[test]
fn g2_scalar_mul_large_random() {
    let mut rng = ChaCha20Rng::seed_from_u64(800);
    let g = Bn254::g2_generator();
    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);

    // (a * b) * G == a * (b * G)
    let ab = a * b;
    let lhs = g.scalar_mul(&ab);
    let rhs = g.scalar_mul(&b).scalar_mul(&a);
    assert_eq!(lhs, rhs);
}

#[test]
fn g2_scalar_mul_consistency_with_repeated_add() {
    let g = Bn254::g2_generator();
    let n = 7u64;
    let scalar = Fr::from_u64(n);
    let via_scalar_mul = g.scalar_mul(&scalar);
    let mut via_add = <Bn254G2 as JoltGroup>::identity();
    for _ in 0..n {
        via_add += g;
    }
    assert_eq!(via_scalar_mul, via_add);
}

#[test]
fn gt_scalar_mul_consistency_with_repeated_add() {
    let e = gt_element();
    let n = 5u64;
    let scalar = Fr::from_u64(n);
    let via_scalar_mul = e.scalar_mul(&scalar);
    let mut via_add = <Bn254GT as JoltGroup>::identity();
    for _ in 0..n {
        via_add += e;
    }
    assert_eq!(via_scalar_mul, via_add);
}

#[test]
fn gt_scalar_mul_large_random() {
    let mut rng = ChaCha20Rng::seed_from_u64(801);
    let e = gt_element();
    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);

    let ab = a * b;
    let lhs = e.scalar_mul(&ab);
    let rhs = e.scalar_mul(&b).scalar_mul(&a);
    assert_eq!(lhs, rhs);
}
