#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
Restricted CCSD implementation which supports both real and complex integrals.
The 4-index integrals are saved on disk entirely (without using any symmetry).
This code is slower than the pyscf.cc.ccsd implementation.

Note MO integrals are treated in chemist's notation

Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)
'''

import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import rintermediates as imd
from pyscf.mp import mp2
from pyscf import __config__

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    #fov = fock[:nocc,nocc:].copy()
    #foo = fock[:nocc,:nocc].copy()
    #fvv = fock[nocc:,nocc:].copy()

    #Foo = imd.cc_Foo(t1,t2,eris)
    #Fvv = imd.cc_Fvv(t1,t2,eris)
    #Fov = imd.cc_Fov(t1,t2,eris)
    #
    ## Move energy terms to the other side
    #Foo[np.diag_indices(nocc)] -= mo_e_o
    #Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation(shape)
    t1new  =fock[:nocc,nocc:].copy()


    # T2 equation
    #eris_ovoo = np.asarray(eris.ovoo, order='C')
    #tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
    #tmp2 += eris.ovvv.conj().transpose(1,3,0,2)
    #tmp = lib.einsum('abic,jc->ijab', tmp2, t1)*0
    #t2new = tmp + tmp.transpose(1,0,3,2)
    #tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    #tmp2 += eris.ovoo.transpose(1,3,0,2).conj()
    #tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    #t2new -= tmp + tmp.transpose(1,0,3,2)
    
    if cc.cc2:
        raise("No CC2")
    else:
        #Loo = imd.Loo(t1, t2, eris)
        #Lvv = imd.Lvv(t1, t2, eris)
        #Loo[np.diag_indices(nocc)] -= mo_e_o
        #Lvv[np.diag_indices(nvir)] -= mo_e_v

        Loo= 2*lib.einsum('klcd,ilcd->ki', eris.ovov.transpose(2,0,3,1), t2)-lib.einsum('lkcd,ilcd->ki', eris.ovov.transpose(2,0,3,1), t2)
        Lvv= 2*lib.einsum('kldc,klda->ac', eris.ovov.transpose(2,0,3,1), t2)-lib.einsum('kldc,lkda->ac', eris.ovov.transpose(2,0,3,1), t2)

        #Woooo = eris.oooo.transpose(2,0,3,1) + lib.einsum('kcld,ijcd->klij', eris.ovov, t2)
        #Wvoov = eris.ovvo.transpose(2,0,3,1) - 0.5*(lib.einsum('ldkc,ilda->akic', eris.ovov, t2) 
        #      + lib.einsum('lckd,ilad->akic', eris.ovov, t2)) + lib.einsum('ldkc,ilad->akic', eris.ovov, t2)
        #Wvovo = eris.oovv.transpose(2,0,3,1) - 0.5*lib.einsum('lckd,ilda->akci', eris.ovov, t2)
        #Wvvvv = eris.vvvv.transpose(2,0,3,1)
        
        Woooo = eris.oooo.transpose(2,0,3,1) + lib.einsum('klcd,ijcd->klij', eris.ovov.transpose(2,0,3,1), t2)
        #Wvoov = eris.ovvo.transpose(2,0,3,1) - 0.5*(lib.einsum('lkdc,ilda->akic', eris.ovov.transpose(2,0,3,1), t2) \
        #      + lib.einsum('lkcd,ilad->akic', eris.ovov.transpose(2,0,3,1), t2)) + lib.einsum('lkdc,ilad->akic', eris.ovov.transpose(2,0,3,1), t2)
        #Wvoov2 = eris.oovv.transpose(2,0,1,3) - 0.5*lib.einsum('lkcd,ilda->akic', eris.ovov.transpose(2,0,3,1), t2)
        Wvoov = eris.ovvo.transpose(3,2,0,1) - 0.5*(lib.einsum('lkdc,ilda->iakc', eris.ovov.transpose(2,0,3,1), t2) \
              + lib.einsum('lkcd,ilad->iakc', eris.ovov.transpose(2,0,3,1), t2)) + lib.einsum('lkdc,ilad->iakc', eris.ovov.transpose(2,0,3,1), t2)
        Wvoov2 = eris.oovv.transpose(1,2,0,3) - 0.5*lib.einsum('lkcd,ilda->iakc', eris.ovov.transpose(2,0,3,1), t2)
        #Wvvvv = eris.vvvv.transpose(2,0,3,1)

        #
        #tau = t2 + np.einsum('ia,jb->ijab', t1, t1)

        t2new = -lib.einsum('ac,ijcb->ijab', Lvv, t2)-lib.einsum('ki,kjab->ijab', Loo, t2) \
            + 2*lib.einsum('iakc,kjcb->ijab', Wvoov, t2)-lib.einsum('iakc,kjcb->ijab', Wvoov2, t2) \
            - lib.einsum('iakc,kjbc->ijab', Wvoov, t2)-lib.einsum('ibkc,kjac->ijab', Wvoov2, t2)
        t2new += t2new.transpose(1,0,3,2)
        
        t2new += lib.einsum('klij,klab->ijab', Woooo, t2)
        t2new += lib.einsum('abcd,ijcd->ijab', eris.vvvv.transpose(2,0,3,1), t2)
     
    t2new += eris.ovov.transpose(0,2,1,3).copy()   

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new *=0.0
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1=None, t2=None, eris=None):
    '''RCCSD correlation energy'''
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    #eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,ijab', t2, eris.ovov.transpose(0,2,1,3))
    e +=  -np.einsum('ijab,ijba', t2, eris.ovov.transpose(0,2,1,3))
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)
    return e.real


class RCCD(ccsd.CCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities

    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if mbpt2:
            pt = mp2.MP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = np.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        return ccsd.CCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)

        elif hasattr(self._scf, 'with_df'):
            logger.warn(self, 'CCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            raise NotImplementedError
            #return _make_df_eris_outcore(self, mo_coeff)

        else:
            return _make_eris_outcore(self, mo_coeff)

    energy = energy
    update_amps = update_amps

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import rccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                rccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
#?        # Note
#?        assert(t1.dtype == np.double)
#?        assert(t2.dtype == np.double)
        return ccsd.CCSD.ccsd_t(self, t1, t2, eris)

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError


class _ChemistsERIs(ccsd._ChemistsERIs):

    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        if slices:
            return self.ovvv[slices]
        else:
            return self.ovvv

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (time.clock(), time.time())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    if callable(ao2mofn):
        eri1 = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), 'f8')
    max_memory = max(MEMORYMIN, mycc.max_memory-lib.current_memory()[0])

    ftmp = lib.H5TmpFile()
    ao2mo.full(mol, mo_coeff, ftmp, max_memory=max_memory, verbose=log)
    eri = ftmp['eri_mo']

    nocc_pair = nocc*(nocc+1)//2
    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    oo = eri[:nocc_pair]
    eris.oooo[:] = ao2mo.restore(1, oo[:,:nocc_pair], nocc)
    oovv = lib.take_2d(oo, tril2sq[:nocc,:nocc].ravel(), tril2sq[nocc:,nocc:].ravel())
    eris.oovv[:] = oovv.reshape(nocc,nocc,nvir,nvir)
    oo = oovv = None

    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    blksize = min(nvir, max(BLKMIN, int(max_memory*1e6/8/nmo**3/2)))
    for p0, p1 in lib.prange(0, nvir, blksize):
        q0, q1 = p0+nocc, p1+nocc
        off0 = q0*(q0+1)//2
        off1 = q1*(q1+1)//2
        buf = lib.unpack_tril(eri[off0:off1])

        tmp = buf[ tril2sq[q0:q1,:nocc] - off0 ]
        eris.ovoo[:,p0:p1] = tmp[:,:,:nocc,:nocc].transpose(1,0,2,3)
        eris.ovvo[:,p0:p1] = tmp[:,:,nocc:,:nocc].transpose(1,0,2,3)
        eris.ovov[:,p0:p1] = tmp[:,:,:nocc,nocc:].transpose(1,0,2,3)
        eris.ovvv[:,p0:p1] = tmp[:,:,nocc:,nocc:].transpose(1,0,2,3)

        tmp = buf[ tril2sq[q0:q1,nocc:q1] - off0 ]
        eris.vvvv[p0:p1,:p1] = tmp[:,:,nocc:,nocc:]
        if p0 > 0:
            eris.vvvv[:p0,p0:p1] = tmp[:,:p0,nocc:,nocc:].transpose(1,0,2,3)
        buf = tmp = None
    log.timer('CCSD integral transformation', *cput0)
    return eris


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf.cc import gccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    #mol.basis = '3-21G'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = RCCSD(mf)
    mycc.max_memory = 0
    eris = mycc.ao2mo()
    emp2, t1, t2 = mycc.init_amps(eris)
    print(lib.finger(t2) - 0.044540097905897198)
    np.random.seed(1)
    t1 = np.random.random(t1.shape)*.1
    t2 = np.random.random(t2.shape)*.1
    t2 = t2 + t2.transpose(1,0,3,2)
    t1, t2 = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1) - 0.25118555558133576)
    print(lib.finger(t2) - 0.02352137419932243)

    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    e, v = mycc.ipccsd(nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    e, v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)


    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-16
    mf.scf()
    mo_coeff = mf.mo_coeff + np.sin(mf.mo_coeff) * .01j
    nao = mo_coeff.shape[0]
    eri = ao2mo.restore(1, mf._eri, nao)
    eri0 = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_coeff.conj(), mo_coeff,
                      mo_coeff.conj(), mo_coeff)

    nocc, nvir = 5, nao-5
    eris = _ChemistsERIs(mol)
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
    eris.fock = np.diag(mf.mo_energy)
    eris.mo_energy = mf.mo_energy

    np.random.seed(1)
    t1 = np.random.random((nocc,nvir)) + np.random.random((nocc,nvir))*.1j - .5
    t2 = np.random.random((nocc,nocc,nvir,nvir)) - .5
    t2 = t2 + np.sin(t2) * .1j
    t2 = t2 + t2.transpose(1,0,3,2)

    mycc = RCCSD(mf)
    t1new_ref, t2new_ref = update_amps(mycc, t1, t2, eris)

    orbspin = np.zeros(nao*2, dtype=int)
    orbspin[1::2] = 1
    eri1 = np.zeros([nao*2]*4, dtype=np.complex)
    eri1[0::2,0::2,0::2,0::2] = \
    eri1[0::2,0::2,1::2,1::2] = \
    eri1[1::2,1::2,0::2,0::2] = \
    eri1[1::2,1::2,1::2,1::2] = eri0
    eri1 = eri1.transpose(0,2,1,3) - eri1.transpose(0,2,3,1)
    erig = gccsd._PhysicistsERIs(mol)
    nocc *= 2
    nvir *= 2
    erig.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    erig.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
    erig.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    erig.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    erig.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    erig.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    erig.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    mo_e = np.array([mf.mo_energy]*2)
    erig.fock = np.diag(mo_e.T.ravel())

    myccg = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
    t1, t2 = myccg.amplitudes_from_ccsd(t1, t2)
    t1new, t2new = gccsd.update_amps(myccg, t1, t2, erig)
    print(abs(t1new[0::2,0::2]-t1new_ref).max())
    t2aa = t2new[0::2,0::2,0::2,0::2]
    t2ab = t2new[0::2,1::2,0::2,1::2]
    print(abs(t2ab-t2new_ref).max())
    print(abs(t2ab-t2ab.transpose(1,0,2,3) - t2aa).max())

